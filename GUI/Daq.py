import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
from nidaqmx.constants import AcquisitionType

plt.ion()

#sampling time & Hz
samp = 20
time = 5


with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan('cDAQ1Mod1/ai0:2')#daq1의 채널
    task.ai_channels.add_ai_accel_chan('cDAQ1Mod2/ai0:2')#daq2의 채널
    task.timing.cfg_samp_clk_timing(rate = samp ,sample_mode = AcquisitionType.FINITE,samps_per_chan=samp*time)
    # 공간할당
    fig = plt.figure()
    a=[]
    space = np.zeros(shape=(samp * time, task.number_of_channels))
    reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
    for i in range(samp*time):
        data = task.read(number_of_samples_per_channel=1)
        a.append(data[0])
        plt.scatter(i,data[2],c='r')
    reader.read_many_sample(space,number_of_samples_per_channel=samp*time)