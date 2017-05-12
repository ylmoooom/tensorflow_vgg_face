from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import myl_vgg_train as mytrain
import myl_vgg_net as mynet


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train_master',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of steps to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")



if __name__ == '__main__':
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  """Train VGG model."""
  global_step = tf.contrib.framework.get_or_create_global_step()
  input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
  input_label = tf.placeholder(tf.float32, [None, FLAGS.num_classes])

  # Build a Graph that computes the logits.
  logits = mynet.inference(input_data, train=True)

  # Calculate loss.
  loss, acc = mytrain.loss(logits, input_label)

  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters.
  train_op = mytrain.train(loss, global_step)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  train_log = open("myl_train_baseline.log", "w")
  test_log = open("myl_test_baseline.log", "w")
  train_log.write("#iter, total_loss, train_acc\n")
  test_log.write("#iter, test_acc\n")
  with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    sess.run(init)
    mynet.load_weights('converted_data', sess)
    print('weights loaded!')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    i = 0
    try:
      while not coord.should_stop():
        # Get images and labels.
        train_images, train_labels = mytrain.train_input('train', 'train.txt')
        summary, _, loss_val, acc_val = sess.run([merged, train_op, loss, acc], feed_dict={input_data: train_images, input_label: train_labels})
        train_writer.add_summary(summary, i)
        print('step #%s, total_loss: %s, train_acc: %s' % (str(i), str(loss_val), str(acc_val)))
        train_log.write(str(i)+', '+str(loss_val)+', '+str(acc_val)+'\n')
        if i % 10 == 0:
          print('testing...testing...testing...')
          test_images, test_labels = mytrain.test_input('test', 'test.txt')
          #top_1 = tf.nn.in_top_k(logits, test_labels, 1)
          #top_5 = tf.nn.in_top_k(logits, tf.argmax(test_labels, 1), 5)
          #pred_top_1 = tf.reduce_mean(tf.cast(top_1, tf.float32))
          #pred_top_5 = tf.reduce_mean(tf.cast(top_5, tf.float32))

          #acc_pred_top_1, acc_pred_top_5 = sess.run([pred_top_1, pred_top_5], feed_dict={input_data: test_images})
          #acc_pred_top_1 = sess.run(pred_top_1, feed_dict={input_data: test_images})
          prediction_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, tf.argmax(test_labels, 1), 1), tf.float32))
          prediction_val = sess.run(prediction_op, feed_dict={input_data: test_images})
          print('test accuracy:')
          print(prediction_val)
          #print('test accuracy@_top_5:')
          #print(acc_pred_top_5)
          test_log.write(str(i)+', '+str(prediction_val)+'\n')
          train_log.flush()
          test_log.flush()
        i += 1
        if i % 500 == 0:
          print('saving...saving...saving...')
          save_path = saver.save(sess, "./train_master/model.ckpt")
          print("Model saved in file: %s" % save_path)
    except Exception, e:
      coord.request_stop(e)
    finally:
      train_writer.close()
      sess.close()
      train_log.close()
      test_log.close()
      coord.request_stop()
      coord.join(threads)
