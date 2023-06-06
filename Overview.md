# 02619 Model Predictive Control

## Learning Objectives and Lecture structure

### Learning Objectives

The Course Learning abjectives are defined as:  

1. Analyze and describe MPC control structures
2. Select processes that can be controlled by MPC
3. Apply convex optimization to estimation, control and system identification
4. Identify a linear model from data
5. Design and tune a linear model based estimator and predictor
6. Design and tune a constrained regulator
7. Synthesize and implement model predictive control systems
8. Test a linear model predictive control system by simulation
9. Implement Nonlinear Model Predictive Controllers
10. Use SQP optimization methods for NMPC
11. Compute solutions and sensitivities in nonlinear ODE models
12. Apply MPC to industrial, biomedical, and financial problems

### Lectures

In addition The course is structured into 13 Lectures

1. Cyber-Physical Systems
   - an overview of MPC and its uses.
2. Modelling and simulation
   - General principals for modelling
3. Case Studies in modeling and simulation
   - Examples of different models, and general pricipels of model building.
4. Modeling of reactive and distributed systems
   - Examples of Distributed systems, modeling using PDE.
5. Realization
   - How to make simulations of the modeled systems
6. State estimation
   - Estimating the underlying state, kalman filter and extended kalman filter
7. Basic MPC
   - Overview of basic concepts of MPC, Objective function and constrained optimization
8. Implementation of MPC
   - Exercise of implementation of the basic MPC
9. Soft MPC and other objective functions
   - Analysis of specific MPC schemes 
10. Tuning, economic MPC, stochastic MPC
    - More specific demonstrations of MPC and demonstrations of how to Tune models
11. NMPC
    - Examples implementation and theory of Nonlinear MPC
12. NMPC 2
    - State estimation and example of NMPC for 4 tank system
13. Case Studies
    - as the name implies

## Exercise and Exam Assignment

The Exam assignment Consist of 14 problems all related to a modified four tank system. As the lecture exercises are sporadic i will mostly be working on the exam assignment.  
For the course i will make 1 report with the solution to exam assignment, in addition to the code used for the assignment.  
I will also try to make a small slide deck explaining important concepts of MPC, and possibly a notebook for illustrations.

Report and slides will be made on overleaf

The 14 problems of the assignment are

1. Control Structure
    - Identify the variables of the system
    - Make block digrams of the MPC problem
    - Relevant Learning objectives: 1,2
    - Relevant Lectures: 2
2. Deterministic and Stochastic Nonlinear Modeling
    - Make mathematical models of the system
      - Deterministic
      - Stochastic
      - Stochastic Differential Equation
    - Relevant Learning Objectives:
    - Relevant Lectures: 3,4,5
3. Nonlinear Simulation - Step Responses
    - Simulate the step responses of the models
    - From step responses calculate transfer function
    - Relevant Learning Objectives: 4,5
    - Relevant Lectures: 3,4
4. Linearization and Discretization.
    - Compute Linearized and Discretised models from Problem 2
    - Analyse models
    - Relevant Learning Objectives: 4,5
    - Relevant Lectures: 3,4
5. State Estimation for the Discrete-time Linear System
    - Design and evaluate static and dynamic (ordinary) kalman filters for the models form problem 3 & 4
    - Relevant Learning Objectives: 5
6. QP Solve interface
    - implementa a solver interface for QP using quadprog or other solvers
    - Relevant Learning Objectives: 3?
    - Relevant lectures: 7.
7. Unconstrained MPC
    - Implement a function for designing an unconstrained MPC based on a Statespace model.
    - Relevant Learning Objectives: 5,6,7
    - Relevant Lectures: 7,8,9
8. Input constrained MPC
    - Implement a function for designing an input constrained MPC based on a Statespace model.
    - Relevant Learning Objectives: 5,6,7
    - Relevant Lectures: 7,8,9
9. MPC with Input Constraints and Soft Output Constraints
    - Implement a function for designing an input and soft constrained MPC based on a Statespace model.
    - Relevant Learning Objectives: 5,6,7
    - Relevant Lectures: 7,8,9
10. Closed-Loop Simulations
    - Make closed-loop simulations of the MPCs
    - Relevant Learning Objectives: 8
    - Relevant Lectures: 7,8,9
11. Nonlinear MPC
    - Design and implement a nonlinear MPC based on stochastic differential equations
    - Relevant Learning Objectives: 9, 10
    - Relevant Lectures: 11, 12
12. Economic Linear MPC and Nonlinear MPC
    - Design and implement a controller that minimize pumping cost
    - Do this both in the linear case and the nonlinear case
    - relevant learning objectives: 9,10
13. PID control
    - Implemt a P, PI, PID-controller, test anbd compare to other controllers
    - Relevant Learning Objectives: -
    - Relevant lectures:
14. Discussion and Conclusion
    - Discuss and conclude on results
    - The discussion should probably be done at the relevant sections

As far as i can see learning objective 11 and 12 are missing from the Assignment
these might need to be done seperately.
