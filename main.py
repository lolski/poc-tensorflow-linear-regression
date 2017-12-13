import csv
import plotly
import plotly.graph_objs as g
import tensorflow as tf
import random

def read_csv_file(path, skip_first_line):
    with open(path) as file:
        reader = csv.reader(file)

        if skip_first_line:
            next(reader, None)

        accumulator = []
        for x, y in reader:
            z = str(float(y) + random.uniform(-15.0, 15.0))
            accumulator.append([x, y, z])

        return accumulator


# ------------------ main ------------------
# step 1: model y = mx + b
y = tf.placeholder(tf.float32, name='y')
m = tf.Variable(0.0, name='slope1')
x = tf.placeholder(tf.float32, name='x')
n = tf.Variable(0.0, name='slope2')
z = tf.placeholder(tf.float32, name='z')
b = tf.Variable(0.0, name='offset')
model_yx = m*x+b
model_yz = n*z+b
current_epoch = tf.placeholder(tf.int32, name='e')

# step 2: read csv
data = read_csv_file('res/train.csv', True)

# step 3: apply gradient descent optimizer in order to find m and b
num_of_epochs = 300
learning_rate=0.0001

loss_fn_yx = tf.square(y - model_yx, name='loss_fn_yx')
loss_fn_yx_with_logs = tf.Print(loss_fn_yx, [current_epoch, loss_fn_yx, m, b])
optimizer_yx = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_fn_yx_with_logs)

loss_fn_yz = tf.square(z - model_yz, name='loss_fn_yz')
loss_fn_yz_with_logs = tf.Print(loss_fn_yz, [current_epoch, loss_fn_yz, n, b])
optimizer_yz = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_fn_yz_with_logs)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    print('training starts...')
    for i in range(num_of_epochs):
        for row in data:
            row_x, row_y, row_z = row
            s.run(optimizer_yx, feed_dict={x: row_x, y: row_y, current_epoch: [i]})
            s.run(optimizer_yz, feed_dict={y: row_y, z: row_z, current_epoch: [i]})
    m_value, n_value, b_value = s.run([m, n, b])

    print('training has ended. m=' + str(m_value) + ', n=' +str(n_value) + ', b='+ str(b_value))

# step 4: plot
data_plot = {
    'x_axis': list(map(lambda e: e[0], data)),
    'y_axis': list(map(lambda e: e[1], data)),
    'z_axis': list(map(lambda e: e[2], data))
}
model_plot = {
    'x_axis': [0, 100],
    'y_axis': [0+b_value, m_value*100 + b_value],
    'z_axis': [0+b_value, n_value*100 + b_value]
}
plotly.offline.plot({
    'data': [
        g.Scatter3d(x=data_plot['x_axis'], y=data_plot['y_axis'], z=data_plot['z_axis'], mode='markers'),
        g.Scatter3d(x=model_yx_plot['x_axis'], y=model_yx_plot['y_axis'], z=model_plot['z_axis'])
    ]
})
