"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from protonet_tf2.protonet import TrainEngine
from protonet_tf2.protonet.models import Prototypical
from protonet_tf2.protonet.datasets import load

def train(config):
    # 初始化多GPU策略
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    np.random.seed(2019)
    tf.random.set_seed(2019)

    # 创建模型和优化器在策略范围内
    with strategy.scope():
        # 模型创建
        model_type = config['model.type']
        now = datetime.now()
        now_as_str = now.strftime('%Y_%m_%d-%H:%M:%S')

        # 模型定义
        n_support = config['data.train_support']
        n_query = config['data.train_query']
        w, h, c = list(map(int, config['model.x_dim'].split(',')))
        model = Prototypical(n_support, n_query, w, h, c,
                           nb_layers=config['model.nb_layers'],
                           nb_filters=config['model.nb_filters'])

        # 优化器
        optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    # 指标定义必须在策略范围外
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_acc = tf.keras.metrics.Mean(name='train_accuracy')
    val_acc = tf.keras.metrics.Mean(name='val_accuracy')

    # 文件路径配置
    model_file = f"{config['model.save_path'].format(model_type, now_as_str)}"
    config_file = f"{config['output.config_path'].format(model_type, now_as_str)}"
    csv_output_file = f"{config['output.train_path'].format(model_type, now_as_str)}"
    train_summary_file = f"{config['summary.save_path'].format('train', model_type, now_as_str)}"
    test_summary_file = f"{config['summary.save_path'].format('test', model_type, now_as_str)}"
    csv_output_map_file = f"results/{config['data.dataset']}/protonet/{config['data.dataset']}_protonet_results.csv"
    summary_file = f"results/summary.csv"

    # 创建目录
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

    # 保存配置文件
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # 初始化CSV记录文件
    with open(csv_output_file, 'w') as f:
        f.write("epoch, loss, accuracy, val_loss, val_accuracy\n")

    # 创建Summary Writer
    train_summary_writer = tf.summary.create_file_writer(train_summary_file)
    val_summary_writer = tf.summary.create_file_writer(test_summary_file)

    # 加载数据集
    data_dir = f"../datasets/{config['data.dataset']}"
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # 定义分布式训练步骤
    @tf.function
    def distributed_train_step(support, query):
        def step_fn(support, query):
            with tf.GradientTape() as tape:
                loss, acc = model(support, query)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss, acc

        per_replica_losses, per_replica_accs = strategy.run(
            step_fn, args=(support, query))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), \
               strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)

    # 定义分布式验证步骤
    @tf.function
    def distributed_test_step(support, query):
        def eval_fn(support, query):
            return model(support, query)

        per_replica_losses, per_replica_accs = strategy.run(
            eval_fn, args=(support, query))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), \
               strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)

    # 创建训练引擎
    train_engine = TrainEngine()

    # 训练过程钩子函数
    def on_start_episode(state):
        if state['total_episode'] % 20 == 0:
            print(f"Episode {state['total_episode']}")
        support, query = state['sample']
        loss, acc = distributed_train_step(support, query)
        train_loss.update_state(loss)
        train_acc.update_state(acc)

    def on_end_episode(state):
        val_loader = state['val_loader']
        for _ in range(config['data.episodes']):
            support, query = val_loader.get_next_episode()
            loss, acc = distributed_test_step(support, query)
            val_loss.update_state(loss)
            val_acc.update_state(acc)

    # 其他钩子函数
    val_losses = []
    min_loss = [100]
    min_loss_acc = [0]

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()

    def on_end_epoch(state):
        epoch = state['epoch']
        template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%'
        print(template.format(
            epoch + 1,
            train_loss.result(),
            train_acc.result() * 100,
            val_loss.result(),
            val_acc.result() * 100
        ))

        # 记录到CSV
        with open(csv_output_file, 'a') as f:
            f.write(f"{epoch + 1}, {train_loss.result():.4f}, {train_acc.result() * 100:.2f}, "
                    f"{val_loss.result():.4f}, {val_acc.result() * 100:.2f}\n")

        # 保存最佳模型
        current_val_loss = val_loss.result().numpy()
        if current_val_loss < min_loss[0]:
            min_loss[0] = current_val_loss
            min_loss_acc[0] = val_acc.result().numpy()
            model.save(model_file)

        # 记录TensorBoard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc.result(), step=epoch)

        # 早停机制
        val_losses.append(current_val_loss)
        if len(val_losses) > config['train.patience'] and \
                all(v >= val_losses[-config['train.patience']] for v in val_losses[-config['train.patience']:]):
            state['early_stopping_triggered'] = True

    # 注册钩子
    train_engine.hooks.update({
        'on_start_episode': on_start_episode,
        'on_end_episode': on_end_episode,
        'on_start_epoch': on_start_epoch,
        'on_end_epoch': on_end_epoch
    })

    # 启动训练
    device_name = 'GPU:0' if config['data.cuda'] else 'CPU:0'
    with tf.device(device_name):
        train_engine.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['train.epochs'],
            n_episodes=config['data.episodes'])

    # 保存最终摘要
    with open(summary_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')},"
                f"{config['data.dataset']},protonet,{config_file},"
                f"{min_loss[0]:.4f},{min_loss_acc[0]:.4f}\n")

    print(f"Training completed. Best validation loss: {min_loss[0]:.4f}, Accuracy: {min_loss_acc[0] * 100:.2f}%")