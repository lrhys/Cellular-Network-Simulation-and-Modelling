# Cellular-Network-Simulation-and-Modelling

## Problem Statement
The telecommunication company XPhone has been receiving complaints from its subscribers regarding quality of service (QoS) along a 40 km long highway connecting two major cities. The highway is covered by its cellular telephony network. The company needs to decide whether or not its system guarantees quality of service (QoS) in terms of percentages of dropped calls and blocked calls. Some measurements have been made of the traffic in the network on the highway. Your task is to model and simulate the system to determine whether the system can meet the quality of service requirements, and if so, which fixed channel allocation scheme offers the best service. 

### Quality of Service (QoS) Requirements
* blocked calls < 2%; and
* dropped calls < 1%

### System Descriptions
The two-way highway is 40 km long. The company uses 20 base stations, each covers a cell with 2 km diameter as shown in the figure below. There is no overlapping of cells. Where the reach of one base station ends, the reach of the next base station starts. Each base station has 10 channels so there are 10 channels available in each cell.

When a subscriber initiates a call from within a cell, a channel in the cell will be allocated to the call. If no free channels are available in the base station, the call is blocked. When a 
subscriber making a call crosses a cell boundary before the end of the 40-km highway, the channel being used in the current cell is released and a new channel in the new cell has to be acquired: this is called a Handover. If a channel is not available in the new base station during a handover the call is dropped. When a subscriber making a call crosses the end of the 40-km highway (either end), the call will be terminated and the channel being used is released.

A Fixed Channel Allocation (FCA) scheme is used. The company wants you to test at least 
two FCA schemes: 
1. No channel reservation
2. 9 channels are allocated to each cell for new calls and handovers and 1 channel is 
reserved for handovers when the other 9 channels are not available. This means a new 
call will not be allocated a channel if there is only one free channel left. 

### The company has provided the following measurements:
1. Call initiation times and their first base stations
2. Call durations 
3. Car speeds. 

### Assumptions
1. The traffic volumes in the two directions are the same. This means the two directions of cars travelling along the highway have equal probabilities. 
2. A car maintains the same speed during a call. 
3. The position of the car initiating a call in a cell is uniformly distributed along the section of the highway covered by the base station.

### Your Tasks
1. Analyze the measured data to find what distributions the inter-arrival times of calls, the locations where calls are generated, the call durations, and car speeds follow respectively. You also need to find the parameter values of these distributions. (The measured data are provided in the file “PCS_TEST_DETERMINSTIC”)

2. Develop a discrete-event simulator.

3. Run your simulator multiple times, each with a warm-up period for different FCA schemes to investigate how handover reservation scheme may affect the quality of service (i.e., blocking and dropping probabilities). Calculate the average values of the percentages of dropped calls and blocked calls and indicate the statistical significance. 

	Answer the following questions: Is the current system able to meet the quality of service requirements and if so how many channels should be reserved for handover for best service? (The percentage of dropped calls is defined as the number of dropped calls divided by the total number of calls; and the percentage of blocked calls is defined as the number of blocked calls divided by the total number of calls.)

### The discrete-event simulator
You may implement the discrete-event simulator using any general purpose programming language (e.g., Python, Java, C, or C++). You may use any PC in the Software Project Laboratory or your own laptop. But you may not use a simulation package or a simulation language. 

Three types of events should be handled by the simulator:
* Call initiation [time, speed, station, position, duration, direction]
* Call termination [time, station]
* Call handover [time, speed, station, duration, direction]

