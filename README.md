# QuCoCi
QuCoCI  is a novel Quantum Community Computing Cyberinfrastructure

QuCuCu includes two primary tasks: 
- containerizing the microservices.
- developing the interface to users.

## Containerized Miroservices
Corresponding to the identified gaps and based on the existing solutions from the team, QuCoCI will containerize:
- Q-VTK: the quantum visualization tookkit for debugging, understanding, and interpretating the quantum codes. Q-VTK includes multiple visualization tools: VACSEN, VENUS, QuantumEyes and Violet
- QEA: the quantum error adaptor that focusing on error prediction and suppression.
- CiCo: The circuit cutting optimization libaray that recommend the optimal circuit cutting solutions on large and complex quantum circuits.

## API and User Inferface
In order to deliver these microservices in a user-friendly way, QuCoCI will future provide interface:
- an API system that disassociates the front-end user interface from the back-end quantum computing resource.
- a service mapper that integrates the existing quantum computing platform to ensure a smoothy circuit “cross-compiling” and codes deployments on different platforms.
- a performance monitoring system that traces the noise variations between different hardware devices
