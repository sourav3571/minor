# 1. First: what is a neural network actually doing?

Forget "AI" for a second.

A neural network is basically a **giant mathematical function with a lot of adjustable numbers**.

For example:

```text
image
  ↓
[ neural network ]
  ↓
0.02 cat
0.91 dog
0.03 horse
0.04 bird
```

The network takes numbers as input and produces numbers as output.

Those internal adjustable numbers are called **weights**.

A tiny neural network might have:

```text
10,000 weights
```

A large one might have:

```text
100,000,000 weights
```

Training is basically:

> **Find values for those weights that make the network produce good answers.**

That's the fundamental idea.

---

# 2. Let's use a stupidly simple example

Imagine we're building a model that predicts whether a number is large.

Input:

```text
x = 7
```

Maybe the model initially does:

```text
7 × 0.2 = 1.4
```

Clearly that's not what we want.

The model has a parameter:

```text
weight = 0.2
```

We want it to learn that maybe:

```text
weight ≈ 1
```

Training repeatedly adjusts these parameters.

Real neural networks just have **millions/billions of these parameters** arranged into layers.

---

# 3. What happens when you train a model?

This is the most important part.

Suppose we have 10,000 cat/dog images.

We give the network an image:

```text
🐕
```

The network initially has random weights.

It might say:

```text
cat: 70%
dog: 30%
```

But the correct answer is:

```text
dog
```

So we calculate:

> "How wrong were you?"

That's called the **loss**.

For example:

```text
loss = 1.8
```

Then mathematics tells us:

> "Change these weights slightly in this direction to reduce the error."

That's **backpropagation + gradient descent**.

Then:

```text
new weights
     ↓
run image again
     ↓
cat: 40%
dog: 60%
     ↓
loss = 0.7
```

Again:

```text
adjust weights
     ↓
run
     ↓
loss = 0.3
```

Eventually:

```text
cat: 2%
dog: 98%
```

That's basically what training is.

---

# 4. Under the hood

The actual loop looks roughly like:

```text
             IMAGE
               │
               ▼
       ┌─────────────────┐
       │ Neural Network  │
       │                 │
       │ weights         │
       │ weights         │
       │ weights         │
       └────────┬────────┘
                │
                ▼
           prediction
                │
                ▼
        compare with answer
                │
                ▼
              LOSS
                │
                ▼
        backpropagation
                │
                ▼
        update weights
                │
                └───────────┐
                            │
                            ▼
                       next image
```

This happens **millions of times**.

---

# 5. So what's a "model architecture"?

Here's a very important distinction.

Imagine:

```text
Architecture:

Input
 ↓
Conv
 ↓
Conv
 ↓
Conv
 ↓
Output
```

This describes **the structure of the network**.

Then:

```text
Weights:

0.183
-0.72
0.004
...
```

describe what the network has **learned**.

So:

### Architecture

= blueprint

### Weights

= learned knowledge

Think of it like a building:

```text
Architecture = building design
Weights      = everything learned/encoded into that building
```

---

# 6. Why can't we just make the model tiny?

Because there's a tradeoff.

Imagine:

```text
Model A
100 million parameters
95% accuracy
```

and:

```text
Model B
1 million parameters
80% accuracy
```

Model A is much more capable, but:

- slower
    
- larger
    
- more memory
    
- more energy
    
- expensive to run
    

Model B is cheap but less accurate.

This becomes particularly painful on **edge devices**.

---

# 7. What is an edge device?

Instead of running AI on:

```text
RTX 5090
128GB RAM
```

imagine running it on:

```text
Raspberry Pi
```

or:

```text
Jetson Nano
```

or:

```text
phone
```

You have limited:

```text
CPU
RAM
storage
power
battery
cooling
```

Therefore you want:

> **The smallest model that is still accurate enough.**

And preferably:

> **The fastest model that consumes the least energy.**

That's the problem your project is attacking.

---

# 8. Where does the Teacher come in?

Suppose we have:

```text
BIG MODEL

ResNet-50
```

It has lots of parameters and is very good.

And:

```text
SMALL MODEL

TinyCNN
```

It is much cheaper but normally performs worse.

We want the big model to **teach** the small model.

That's Knowledge Distillation.

---

# 9. Normal training

Normally the student sees:

```text
Image → Student → Answer
```

Suppose the correct answer is:

```text
DOG
```

So:

```text
Student:

cat   0.1
dog   0.8
horse 0.1
```

We compare it with:

```text
Correct = dog
```

---

# 10. Distillation

Now we have:

```text
Image
  │
  ├─────────────► Teacher
  │                  │
  │                  ▼
  │             cat  10%
  │             dog  80%
  │             horse 10%
  │
  └─────────────► Student
                       │
                       ▼
                  cat 20%
                  dog 60%
                  horse 20%
```

The student isn't merely told:

> "The answer is dog."

It also gets:

> "The teacher thinks it's 80% dog, but there's a 10% possibility of cat and horse."

That extra information can help the small model learn better.

---

# 11. Now we arrive at NAS

Here's the big idea.

Normally **you choose the student architecture**.

You might write:

```python
student = MobileNet(...)
```

But what if we don't know which architecture is best?

Maybe:

```text
Architecture A → 90% accuracy
Architecture B → 92%
Architecture C → 91%
Architecture D → 94%
Architecture E → 89%
```

Instead of manually designing them, we create a system that **searches for them automatically**.

That's:

# Neural Architecture Search

NAS.

---

# 12. NAS is basically an automated architect

Imagine giving an AI this toolbox:

```text
Operations:

Conv 3×3
Conv 5×5
Depthwise Conv
Pooling
Skip connection
```

And rules:

```text
layers = 4–12
channels = 16–64
```

It can create:

```text
Model 1

Conv3
 ↓
Conv3
 ↓
Pool
 ↓
Conv1
```

Then:

```text
Model 2

DWConv3
 ↓
Conv1
 ↓
DWConv5
 ↓
Skip
```

Then:

```text
Model 3

Conv5
 ↓
DWConv3
 ↓
DWConv3
 ↓
Conv1
```

and so on.

It searches for good designs.

---

# 13. But searching only for accuracy isn't enough

Suppose NAS discovers:

```text
Model A

Accuracy = 98%
Size = 200MB
Latency = 500ms
```

Congratulations...

It is useless on a Raspberry Pi for real-time inference.

Instead:

```text
Model B

Accuracy = 94%
Size = 8MB
Latency = 15ms
Energy = low
```

might be vastly better.

So we have multiple things we care about.

---

# 14. Your project is essentially this optimization problem

We want:

```text
                 MODEL
                   │
       ┌───────────┼────────────┐
       ▼           ▼            ▼
   Accuracy      Speed        Size
       │           │            │
       └───────────┼────────────┘
                   ▼
                 Energy
```

We want:

```text
Accuracy   → HIGH
Latency    → LOW
Energy     → LOW
Size       → LOW
```

And that's **multi-objective optimization**.

---

# 15. Why can't we just say:

```text
accuracy - latency - size - energy
```

?

Because the units are completely different.

For example:

```text
accuracy = 92%
latency = 17 ms
size = 4 MB
energy = 0.2 J
```

You can't meaningfully add those directly.

So you normalize them and/or use weights.

Conceptually:

```text
score =
    0.5 × accuracy
  + 0.2 × speed
  + 0.2 × energy_efficiency
  + 0.1 × size_efficiency
```

But the weights can change depending on what you care about.

---

# 16. That's where "adaptive" comes from

Imagine:

### Camera system

You care enormously about real-time performance.

```text
Accuracy = 40%
Latency  = 40%
Energy   = 15%
Size     = 5%
```

But another application might say:

> "I don't care if inference takes 100ms. I need maximum accuracy."

Then:

```text
Accuracy = 70%
Latency  = 10%
Energy   = 10%
Size     = 10%
```

Your optimizer can adapt to those requirements.

---

# 17. And now the hardware-aware part

This is a REALLY important detail.

Suppose:

```text
Model A
10M parameters
```

and:

```text
Model B
15M parameters
```

You might assume:

```text
A = faster
```

Not necessarily.

Why?

Because actual hardware performance depends on:

- memory access
    
- CPU/GPU architecture
    
- supported operations
    
- cache
    
- kernel implementations
    
- framework
    
- quantization
    
- hardware acceleration
    

For example:

```text
Model A
10M parameters
but weird operations
        ↓
Raspberry Pi
        ↓
30 ms
```

while:

```text
Model B
15M parameters
simple optimized operations
        ↓
Raspberry Pi
        ↓
18 ms
```

So your project should eventually measure **real hardware performance**, not just theoretical FLOPs.

---

# 18. Put everything together

Now the entire project makes sense.

We start with:

```text
                    BIG TEACHER
                   ResNet-18/50
                        │
                        │ teaches
                        ▼
              ┌─────────────────────┐
              │ Architecture Search │
              └──────────┬──────────┘
                         │
              generates student models
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Model A     Model B     Model C
             │           │           │
             └───────────┼───────────┘
                         │
                    train with
                   teacher guidance
                         │
                         ▼
                     Evaluate
                         │
       ┌─────────────────┼────────────────┐
       ▼                 ▼                ▼
   Accuracy           Latency            Size
                                           │
                                         Energy
       └─────────────────┼────────────────┘
                         ▼
                Multi-objective
                   optimizer
                         │
                         ▼
                  Best candidates
                         │
                         ▼
                 Raspberry Pi
                         │
                         ▼
               REAL measurements
                         │
                         ▼
                  Feed results
                    back into
                     search
                         │
                         └───────────► repeat
```

**That loop is your project.**

---

# 19. What exactly are we trying to achieve?

The goal isn't:

> "Create the most accurate neural network."

And it isn't:

> "Create the smallest neural network."

It's:

> **Automatically discover neural networks that provide an excellent trade-off between accuracy, speed, energy consumption, and size on edge hardware.**

For example, you might ultimately discover:

```text
                   Your discovered model

Accuracy     ████████████████████  93.2%

Latency      ███                    14 ms

Size         ████                   4.2 MB

Energy       ███                    0.08 J
```

while the existing model is:

```text
Existing model

Accuracy     █████████████████████  94.0%

Latency      █████████████████      85 ms

Size         █████████████████      35 MB

Energy       ███████████████        0.5 J
```

You've sacrificed only:

```text
0.8% accuracy
```

but gained:

```text
~6× lower latency
~8× smaller model
~6× lower energy
```

**That is a meaningful result.**

---

# 20. What "probability density" has to do with this

Don't worry about probability density yet.

You don't need probability theory to understand the core of this project.

There are a few ML concepts you should learn **in this order**:

```text
1. What is a tensor?
        ↓
2. Neural network layers
        ↓
3. Forward pass
        ↓
4. Loss function
        ↓
5. Gradient
        ↓
6. Backpropagation
        ↓
7. Gradient descent
        ↓
8. Training loop
        ↓
9. Classification probabilities
        ↓
10. Softmax
        ↓
11. Knowledge distillation
        ↓
12. CNNs
        ↓
13. Model compression
        ↓
14. NAS
        ↓
15. Multi-objective optimization
```

**Probability density comes much later**, and honestly, for the first implementation of this project, you don't need deep probability theory.

---

# 21. One analogy I want you to remember

Imagine you're designing cars.

You have:

### Teacher

A Ferrari.

Very powerful.

```text
Fast
Expensive
Heavy
High fuel consumption
```

You want to build a small city car.

You have thousands of possible designs:

```text
engine size
gearbox
wheel size
body
aerodynamics
weight
etc.
```

You want the computer to automatically search for a design that:

```text
        maximize
          safety
            ↑
            │
       ┌────┴────┐
       │         │
     speed      range
       │         │
       └────┬────┘
            │
        minimize
     cost / weight
```

The Ferrari teaches the small car how to perform well.

The search algorithm experiments with designs.

The real road test tells you how fast the car actually is.

Then the algorithm creates better designs.

**Your project does exactly this, except the "cars" are neural networks.**

---

# 22. The most important distinction for you

As a software engineer, I want you to separate these three things in your head:

```text
             NEURAL NETWORK
                  │
        ┌─────────┴─────────┐
        │                   │
   Architecture           Weights
        │                   │
   "How it's built"     "What it learned"
        │                   │
        │                   │
        └─────────┬─────────┘
                  │
                Model
```

**Training** finds good weights.

**NAS** searches for good architectures.

**Knowledge distillation** helps a smaller architecture learn from a larger model.

**Multi-objective optimization** decides which architectures are good when you care about several competing goals.

**Hardware-aware optimization** makes sure "good on paper" actually means "good on the target machine."

And **your research project combines all five.**

---

## If I were mentoring you through this

I would **not** start by reading NAS papers.

That would be like trying to understand Kubernetes before knowing what a process is.

I'd first make you build this tiny thing yourself:

```text
Python + PyTorch

        ↓

Load CIFAR-10

        ↓

Build tiny neural network

        ↓

Forward pass

        ↓

Calculate loss

        ↓

Backpropagation

        ↓

Optimizer updates weights

        ↓

Watch accuracy improve
```

Once you've actually watched:

```text
epoch 1 → 35%
epoch 2 → 51%
epoch 5 → 72%
epoch 10 → 84%
```

and understand **why the weights are changing**, terms like _teacher, student, distillation, NAS, logits, softmax, search space, Pareto frontier_ stop being magic vocabulary.

Then we'd build **teacher → student distillation**, then **architecture search**, and only after that add the fancy adaptive hardware optimization.

That is the path I'd recommend for you as a programmer entering ML: **understand the machine first, then the research framework.**