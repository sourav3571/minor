
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