import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def main(data_x):
    x = tf.placeholder(shape=[len(data_x)], dtype=tf.float32, name="x")

    func = tf.reduce_mean(x * x * x)
    gradients = tf.gradients(func, x)
    hessians = tf.hessians(func, x)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    func_value = sess.run(func,
                          feed_dict={
                              x: data_x
                          }
                          )
    gradients_value = sess.run(gradients,
                               feed_dict={
                                   x: data_x
                               }
                               )
    hessians_value = sess.run(hessians,
                              feed_dict={
                                  x: data_x
                              }
                              )
    return func_value, gradients_value, hessians_value


print(*main([-10., 20., 30.]), sep="\n")
