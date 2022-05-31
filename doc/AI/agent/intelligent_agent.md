### What is an Agent?

An agent can be anything that perceiveits environment through sensors and act upon environment through actuators. An Agent runs in the cycle of **perceiving**, **thinking**, and **acting**.

An agent can be:
- **Human-Agent**: A human agent has eyes, ears and other organs which owrk for sensors and hand, legs, vocal tract work for actuators.
- **Robotic Agent**: A robotic agent can have cameras, infrared range finder, NLP for sensors and various motors for actuators.
- **Software Agent**: Software agent can have keystrokes, file contents as snesory input and act on those inputs and display output on the screen.

The world has so many agents such as theromstat, mobile phone, camera, even human itself.

- **Sensor**: Sensor is a device which detects the change in the environm,ent and sends the information to other electronic devices. An agent observes its environment through sensors.
- **Actuators**: Actuators are the component of machines that converts energy into motion. The actuators are only responsible for moving and controlling a system. An actuator can be an electric moter, gears, rails, etc.
- **Effectors**: Effectors are the devices which affect the environment. Effectors can be legs, wheels, arms, fingers, wings, fins, and display screen.

### Intelligent Agents

An intelligent agent is an autonomous entity which act upon an environment using sensors and actuators for achieving goals. An intelligent agent may learn from the environment to achieve their goals. A thermostat is an exmaple of an intelligent agent.

Following are the main four rules for an AI agent:
- **Rule 1**：An AI agent must have the ability to perceive the environment.
- **Rule 2**：The observation must be used to make decisions.
- **Rule 3**：Decisions should result in action.
- **Rule 4**：The action taken by an AI agent must be a rational action.

> Rational agents in AI are very similar to intelligent agents.

### Rational Agents

An rational agent is an agent which has clear preference, models uncertainty, and acts in a way to maximize its performance measure with all possible actions.

A rational agent is said to perform the right things. AI is about creating rational agents to use for game theory and decision theory for various real-world scenarios.

For an AI agent, the rational action is most important because in AI reinforcement learning algorithm, for each best possible action, agent gets the possible reward and for each wrong action, an agent gets a negative reward.

#### Rationality

The rationality of an agent is measured by its performance measure. Rationality can be judged on the basis of following points:
- Performance measure which defines the success criterion.
- Agent prior knowledge of its environment.
- Best possible actions that an agent can perform.
- The sequence of percepts.

### Structure of an AI agent

The task of AI is to design an agent program which implements the agent function. The structure of an intelligent agent is a combination of architecture and agent program. It can be viewed as:

> Agent = Architecture + Agent Program

Following are the main three terms involved in the structure of an AI agent:

- **Architecture**: Architecture is machinery that an AI agent exectues on.
- **Agent function**: Agent function is used to map a percept to an action.
- **Agent program**: Agent program is an implement of agent function. An agent program executes on the physical architecture to produce function f.

#### PEAS Representation

PEAS is a type of model on which an AI agent works upon. When we define an AI agent or raitonal agent, then we can group its properties under PEAS representation model. It is made up of four words:
- **P**: Performance measure
- **E**: Environment
- **A**: Actuators
- **S**: Sensors

E.g. PEAS for self-driving cars
- **Performance**: Safety, time, legal drive, comfort
- **Environment**: Roads, other vehicles, road signs, pedestrain
- **Actuators**: Steering, accelerator, brake, signal, horn
- **Sensors**: Camera, GPS, speedometer, odometer, accelerometer, sonar

E.g. Some agents with their PEAS representation

| **Agent**          | **Performance measure**                             | **Environment**                                          | **Actuators**                         | **Sensors**                                                             |
| ------------------ | --------------------------------------------------- |--------------------------------------------------------- | ------------------------------------- |------------------------------------------------------------------------ |
| Medical Diagnose   | Healthy patient<br>Minimized cost                   | Patient<br>Hospitcal, Staff                              | Test<br>Treatments                    | Keyboard (Entry of symptoms)                                            |
| Vacuum Cleaner     | Cleanness<br>Efficiency<br>Battery life<br>Security | Room, Table<br>Wood floor<br>Carpet<br>Various obstacles | Wheels<br>Brushes<br>Vaccum Extractor | Camera<br>Dirt detection sensor<br>Cliff Sensor<br>Infrared Wall Sensor |
| Part-picking Robot | Percentage of parts in correct bins                 | Conveyor bet with parts<br>Bins                          | Jointed Arms<br>Hand                  | Camera<br>Joint angle sensors                                           |

