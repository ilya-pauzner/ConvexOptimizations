import tensorflow.compat.v1 as tf
import functools
# tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)


tf.disable_v2_behavior()

@functools.lru_cache()
def count(data_x, function):
    x = tf.placeholder(shape=[len(data_x)], dtype=tf.float32, name="x")

    func = function(x)
    gradients = tf.gradients(func, x)
    hessians = tf.hessians(func, x)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    return sess.run([func, gradients, hessians], feed_dict={x: data_x})


if __name__ == '__main__':
    print(*count([-10., 20., 30.], lambda x: tf.reduce_sum(x * x * x) / 3), sep="\n")
