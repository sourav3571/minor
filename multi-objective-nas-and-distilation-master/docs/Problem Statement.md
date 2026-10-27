# Title:- Adaptive Teacher-Guided Multi-Objective Neural Architecture Search for Lightweight Edge Intelligence

Most modern deep learning models are far too heavy and power-hungry to run directly on low-power devices like smartphones, microcontrollers, or smart cameras. Designing smaller models manually requires endless trial and error. This project creates an automated system that builds compact, high-performance neural networks on its own. It uses a smart "teacher" model to guide smaller "student" models during training. Instead of only looking at accuracy, our system balances four critical goals at once: high accuracy, low execution latency, low power consumption, and small memory footprint. This allows us to generate models ready for real-time deployment on hardware like Raspberry Pi and NVIDIA Jetson.

## objective

- Build an automated search framework (NAS) that generates compact neural network architectures suited for resource-constrained hardware.
    
- Integrate adaptive Knowledge Distillation so lightweight student models can learn effectively from larger teacher networks.
    
- Optimize four key performance metrics simultaneously: maximize accuracy while minimizing hardware latency, energy draw, and model size.
    
- Deploy and measure the generated models on physical hardware boards (Raspberry Pi / NVIDIA Jetson) using standard CIFAR-10 and CIFAR-100 benchmarks.
    
- Produce a research paper detailing the results for conference submission.

## description
- Teacher Setup & Search Space: We start by training a large, high-accuracy teacher model (like ResNet or EfficientNet) on CIFAR datasets. We then define a pool of lightweight architectural building blocks (such as depthwise convolutions and flexible layer widths) for the candidate student models.
    
- Multi-Objective Search: We use an evolutionary optimization algorithm (such as NSGA-II) to test different student combinations. The algorithm scores candidates based on a combined metric of accuracy, speed, energy, and memory size.
    
- Adaptive Teacher Guidance: As student architectures are generated, we train them using Knowledge Distillation. The guidance strength from the teacher model adjusts dynamically based on how small or complex the current student network is.
    

- Hardware Benchmarking: We export the best student models using ONNX and TensorRT, run them directly on target edge hardware (Raspberry Pi / Jetson Nano), and log real-time inference latency (milliseconds) and energy consumption.
