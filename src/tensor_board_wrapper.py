import os, sys, platform,time, subprocess, datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def mk_dir(_dir):
    try:
        os.mkdir(_dir)
    except OSError:
        pass

class TensorBoardWrapper(object):
    def __init__(self, rm_old_logs=False):
        super().__init__()
        self.rm_old_logs = rm_old_logs
        logs_path = './tb_logs'
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir = os.path.join(logs_path,self.time_stamp)
        mk_dir(self.logdir)
        self.is_linux = False
        if platform.system()=="Linux":
            self.is_linux = True  

        self.get_file_writer()
    
    def get_file_writer(self):
        if self.rm_old_logs:
            self.rm_recursive(self.logdir) 
        self.file_writer = tf.summary.create_file_writer(self.logdir)
        self.file_writer.set_as_default()

    def add_scalar(self, name, data, step=None, description=None):
        with self.file_writer.as_default():
            tf.summary.scalar(name, data, step=step, description=description)
        self.file_writer.flush()

    def add_image(self, name, data, step=None, max_outputs=3, description=None):
        with self.file_writer.as_default():
            if type(data).__module__ == np.__name__:
                if len(np.shape(data))==4:
                    tf.summary.image(name, data, step=step,max_outputs=max_outputs, description=description)
                else:
                    raise Exception(f"Data has {len(np.shape(data))} but it must have 4 dimensions")
            else:
                raise Exception("Data must be of type numpy array")
        self.file_writer.flush()

    def generate_model_graph(self,model, sample_input):
        @tf.function
        def traceme(x):
            return model(x)
        tf.summary.trace_on(graph=True, profiler=True)
        traceme(sample_input)
        with self.file_writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=self.logdir)
        self.file_writer.flush()

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def launch_tensorboard(self, port=6006):
        if self.is_linux:
            subprocess.Popen("sudo killall tensorboard", shell=True)
        command = "tensorboard --logdir="+ self.logdir + " --host 0.0.0.0 --port "+ str(port)
        subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        print("tensorboard launched at 0.0.0.0:"+ str(port))
