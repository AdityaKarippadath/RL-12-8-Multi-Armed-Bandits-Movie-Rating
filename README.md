# 🎬 MovieLens Multi-Armed Bandit Recommendation System

A comprehensive implementation of **Multi-Armed Bandit algorithms for movie recommendation** using the **MovieLens 1M dataset**.

This project demonstrates how online decision-making algorithms can learn which movies are likely to generate positive user feedback while balancing the fundamental **exploration-exploitation trade-off**.

The project implements and compares:

* Random Baseline
* Epsilon-Greedy
* Decaying Epsilon-Greedy
* Upper Confidence Bound (UCB)
* Thompson Sampling

The algorithms are evaluated using:

* Total Reward
* Average Reward
* Cumulative Reward
* Cumulative Regret
* Movie Selection Frequency
* Multiple Independent Runs

> **Important:** MovieLens 1M is an offline historical dataset. This project therefore constructs a simulated Bernoulli bandit environment from historical movie ratings. It is an educational/offline simulation rather than a real-time recommendation system.

---

# 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Objectives](#-objectives)
* [Dataset](#-dataset)
* [Dataset Structure](#-dataset-structure)
* [Recommendation Problem Formulation](#-recommendation-problem-formulation)
* [Reward Definition](#-reward-definition)
* [Bandit Environment](#-bandit-environment)
* [Algorithms](#-algorithms)

  * [Random Baseline](#1-random-baseline)
  * [Epsilon-Greedy](#2-epsilon-greedy)
  * [Decaying Epsilon-Greedy](#3-decaying-epsilon-greedy)
  * [UCB](#4-upper-confidence-bound-ucb)
  * [Thompson Sampling](#5-thompson-sampling)
* [Exploration vs Exploitation](#-exploration-vs-exploitation)
* [Project Architecture](#-project-architecture)
* [Project Structure](#-project-structure)
* [Technologies Used](#-technologies-used)
* [Installation](#-installation)
* [Running on Google Colab](#-running-on-google-colab)
* [Running Locally](#-running-locally)
* [Implementation Workflow](#-implementation-workflow)
* [Evaluation Metrics](#-evaluation-metrics)
* [Experiments](#-experiments)
* [Results](#-results)
* [Limitations](#-limitations)
* [Important Offline Evaluation Note](#-important-offline-evaluation-note)
* [Future Improvements](#-future-improvements)
* [Contextual Bandit Extension](#-contextual-bandit-extension)
* [Real-World Applications](#-real-world-applications)
* [Learning Outcomes](#-learning-outcomes)
* [Conclusion](#-conclusion)
* [Author](#-author)
* [License](#-license)

---

# 🎯 Project Overview

Traditional recommendation systems often attempt to predict which items a user will like based on historical data.

A Multi-Armed Bandit approaches the problem differently.

Instead of only predicting a rating, the system repeatedly makes decisions:

```text
        User
          │
          ▼
    Select Movie
          │
          ▼
    Receive Reward
          │
          ▼
     Learn From Reward
          │
          ▼
    Select Next Movie
          │
          ▼
        ...
```

The system must learn which actions provide high rewards while still exploring alternatives that may perform better.

In this project, each movie is treated as a **bandit arm**.

For example:

```text
Arm 0 → Toy Story
Arm 1 → Jumanji
Arm 2 → Grumpier Old Men
Arm 3 → Waiting to Exhale
...
```

The algorithm does not initially know which movie has the highest probability of receiving a positive reward.

---

# 🧠 What is a Multi-Armed Bandit?

The Multi-Armed Bandit problem is a classic sequential decision-making problem.

Imagine several slot machines:

```text
             Multi-Armed Bandit

        ┌──────────┐
        │  Arm 0   │
        └──────────┘
             │
           Reward

        ┌──────────┐
        │  Arm 1   │
        └──────────┘
             │
           Reward

        ┌──────────┐
        │  Arm 2   │
        └──────────┘
             │
           Reward

             ...

        ┌──────────┐
        │  Arm N   │
        └──────────┘
             │
           Reward
```

Each arm has an unknown reward distribution.

The goal is to discover the best arm while maximizing the total reward obtained during the learning process.

---

# 🎯 Problem Statement

Given a collection of movies, determine which movie is most likely to produce a positive user response while minimizing the reward lost during exploration.

Formally, for each arm (a), there is an unknown expected reward:

[
\mu_a = E[R|A=a]
]

The optimal arm is:

[
a^* = \arg\max_a \mu_a
]

The algorithm must learn this through repeated interactions.

---

# 🎯 Objectives

The main objectives of this project are:

1. Load and preprocess the MovieLens 1M dataset.
2. Analyze movie ratings.
3. Convert movie ratings into binary rewards.
4. Treat movies as bandit arms.
5. Construct a simulated Bernoulli bandit environment.
6. Implement multiple MAB algorithms from scratch.
7. Compare their learning behavior.
8. Measure cumulative reward.
9. Measure cumulative regret.
10. Analyze movie selection frequency.
11. Perform multiple independent experiments.
12. Compare the algorithms statistically through mean performance.
13. Explore how exploration parameters affect performance.
14. Provide a foundation for extending the system to Contextual Bandits.

---

# 📊 Dataset

This project uses the **MovieLens 1M dataset**.

MovieLens 1M contains approximately:

* 1 million ratings
* 6,000 users
* 4,000 movies

The dataset contains three primary files:

```text
ml-1m/
│
├── ratings.dat
├── movies.dat
├── users.dat
└── README
```

The main file used to construct the bandit reward environment is:

```text
ratings.dat
```

---

# 📁 Dataset Structure

## `ratings.dat`

The ratings file contains:

```text
UserID::MovieID::Rating::Timestamp
```

Example:

```text
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
1::3408::4::978300275
```

The columns are:

| Column      | Description             |
| ----------- | ----------------------- |
| `UserID`    | Unique user identifier  |
| `MovieID`   | Unique movie identifier |
| `Rating`    | Rating from 1 to 5      |
| `Timestamp` | Rating timestamp        |

---

## `movies.dat`

Format:

```text
MovieID::Title::Genres
```

Example:

```text
1::Toy Story (1995)::Animation|Children's|Comedy
```

Columns:

| Column    | Description             |
| --------- | ----------------------- |
| `MovieID` | Unique movie identifier |
| `Title`   | Movie title             |
| `Genres`  | Movie genres            |

---

## `users.dat`

Format:

```text
UserID::Gender::Age::Occupation::Zip-code
```

Columns:

| Column       | Description            |
| ------------ | ---------------------- |
| `UserID`     | Unique user identifier |
| `Gender`     | User gender            |
| `Age`        | Age category           |
| `Occupation` | Occupation code        |
| `Zip-code`   | ZIP code               |

The current implementation primarily uses `ratings.dat` and `movies.dat`.

The user information is reserved for future **Contextual Bandit** extensions.

---

# 🔄 Recommendation Problem Formulation

The recommendation problem is transformed into a Multi-Armed Bandit problem.

## Traditional recommendation view

```text
User
  │
  ▼
Predict Movie Rating
  │
  ▼
Recommend Movie
```

## Multi-Armed Bandit view

```text
User
  │
  ▼
Choose Movie / Arm
  │
  ▼
Observe Reward
  │
  ▼
Update Knowledge
  │
  ▼
Choose Next Movie
```

Each movie is treated as an arm.

```text
Movie A → Arm 0
Movie B → Arm 1
Movie C → Arm 2
...
Movie N → Arm N
```

---

# 🎁 Reward Definition

The original MovieLens ratings range from 1 to 5.

For the Bernoulli Bandit implementation, ratings are converted into binary rewards.

The transformation is:

```text
Rating 1 → Reward 0
Rating 2 → Reward 0
Rating 3 → Reward 0
Rating 4 → Reward 1
Rating 5 → Reward 1
```

Therefore:

[
Reward =
\begin{cases}
1 & \text{if rating} \ge 4 \
0 & \text{if rating} < 4
\end{cases}
]

This makes the reward compatible with Bernoulli Bandit algorithms.

---

# 📈 Movie Reward Probability

For every movie, we calculate its historical positive-rating probability:

[
P(Reward=1|Movie)
=================

\frac{\text{Positive Ratings}}
{\text{Total Ratings}}
]

For example:

```text
Movie A

Total ratings = 1000
Ratings >= 4 = 700

Positive reward probability:

700 / 1000 = 0.70
```

This probability is used by the simulator to generate rewards.

---

# 🎰 Bandit Environment

The project creates a simulated Bernoulli Bandit environment.

Suppose the historical data gives:

```text
Movie A → 0.20
Movie B → 0.50
Movie C → 0.70
Movie D → 0.40
Movie E → 0.30
```

The environment behaves approximately like:

```text
Select Movie C
       ↓
Probability of reward = 0.70
       ↓
Random sample
       ↓
Reward = 1 or 0
```

The bandit algorithm does **not** receive the true probabilities.

It only receives:

```text
Movie selected
      ↓
Reward
```

and must learn from those observations.

---

# 🎯 Number of Arms

MovieLens contains thousands of movies.

Using every movie as an arm can make the basic experiment unnecessarily large.

Therefore, the notebook uses the:

```text
Top 100 movies by number of historical ratings
```

as the initial set of arms.

This provides a sufficiently large but computationally manageable experiment.

The number of arms can be changed:

```python
N_ARMS = 100
```

For example:

```python
N_ARMS = 50
```

or:

```python
N_ARMS = 500
```

---

# 🤖 Algorithms

The project implements five approaches.

1. Random Baseline
2. Epsilon-Greedy
3. Decaying Epsilon-Greedy
4. Upper Confidence Bound
5. Thompson Sampling

---

# 1. Random Baseline

The random baseline selects a movie completely randomly.

```text
Movie A → 20%
Movie B → 20%
Movie C → 20%
Movie D → 20%
Movie E → 20%
```

It does not learn from rewards.

This provides a baseline against which intelligent algorithms can be compared.

### Advantages

* Extremely simple.
* No parameters.
* Useful as a baseline.

### Disadvantages

* Never learns.
* Continues selecting poor movies.
* Usually has high regret.

---

# 2. Epsilon-Greedy

Epsilon-Greedy uses a parameter:

[
\epsilon
]

With probability (\epsilon):

```text
Explore
```

With probability (1-\epsilon):

```text
Exploit
```

For:

```text
epsilon = 0.1
```

approximately:

```text
10% → Random exploration
90% → Best-known movie
```

The decision rule is:

[
A_t =
\begin{cases}
\text{random arm} & \text{with probability } \epsilon\
\arg\max_a Q(a) & \text{otherwise}
\end{cases}
]

---

## Reward Update

Each arm maintains an estimated value:

[
Q(a)
]

The estimate is updated using:

[
Q(a)
\leftarrow
Q(a)
+
\frac{R-Q(a)}
{N(a)}
]

where:

* (R) = observed reward
* (N(a)) = number of selections of arm (a)
* (Q(a)) = estimated reward

---

# 3. Decaying Epsilon-Greedy

Fixed epsilon continues exploring at the same rate throughout the experiment.

Decaying Epsilon-Greedy reduces epsilon over time.

Initially:

```text
High epsilon
     ↓
High exploration
```

Later:

```text
Low epsilon
     ↓
High exploitation
```

The notebook uses:

```python
initial_epsilon = 1.0
min_epsilon = 0.01
decay = 0.999
```

The update is:

[
\epsilon_t =
\max(
\epsilon_{min},
\epsilon_t \times decay
)
]

This allows the algorithm to learn broadly during the early stage and increasingly exploit its knowledge later.

---

# 4. Upper Confidence Bound (UCB)

UCB considers both:

* Estimated reward
* Uncertainty

The formula is:

[
UCB(a)
======

Q(a)
+
c
\sqrt{
\frac{\ln(t)}
{N(a)}
}
]

where:

* (Q(a)) = estimated reward
* (c) = exploration coefficient
* (t) = timestep
* (N(a)) = number of times arm (a) was selected

The second term is the exploration bonus.

---

## UCB Intuition

If a movie has:

```text
High estimated reward
+
High uncertainty
```

UCB may select it.

If a movie has:

```text
High estimated reward
+
Low uncertainty
```

its exploration bonus becomes smaller.

This produces more targeted exploration than randomly choosing an arm.

---

# 5. Thompson Sampling

Thompson Sampling is a Bayesian approach.

For Bernoulli rewards, each arm is represented using a Beta distribution:

[
Beta(\alpha,\beta)
]

Initially:

[
\alpha=1,\quad\beta=1
]

This represents a uniform prior.

At each timestep:

1. Sample a value from every arm's Beta distribution.
2. Select the arm with the largest sample.
3. Observe the reward.
4. Update the selected arm.

If:

```text
Reward = 1
```

then:

[
\alpha \leftarrow \alpha+1
]

If:

```text
Reward = 0
```

then:

[
\beta \leftarrow \beta+1
]

---

# ⚖️ Exploration vs Exploitation

The central challenge of the project is:

```text
             Exploration
                  ↕
             Exploitation
```

## Exploration

Try movies that are uncertain.

Purpose:

> Discover potentially better movies.

## Exploitation

Recommend movies that currently appear to perform well.

Purpose:

> Maximize immediate reward.

---

# 🏗️ Project Architecture

The complete pipeline is:

```text
                  MovieLens 1M
                       │
                       ▼
                Load Dataset
                       │
                       ▼
              Data Preprocessing
                       │
                       ▼
              Rating → Reward
                       │
                       ▼
             Calculate Movie Stats
                       │
                       ▼
                Select Top Arms
                       │
                       ▼
             Simulated Environment
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Epsilon        UCB       Thompson
       Greedy                  Sampling
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Receive Reward
                       │
                       ▼
                 Update Model
                       │
                       ▼
                  Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Reward       Regret       Selection
                                   Frequency
```

---

# 📂 Project Structure

Recommended GitHub structure:

```text
movielens-multi-armed-bandit/
│
├── notebooks/
│   └── movielens_multi_armed_bandit.ipynb
│
├── data/
│   └── README.md
│
├── results/
│   ├── cumulative_reward.png
│   ├── cumulative_regret.png
│   ├── learning_curves.png
│   └── algorithm_comparison.png
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

### Important

Do **not** commit the MovieLens dataset if its license/terms do not permit redistribution.

Instead, provide instructions for downloading it.

---

# 🛠️ Technologies Used

## Programming Language

* Python 3

## Libraries

* NumPy
* Pandas
* Matplotlib

## Development Environment

* Google Colab
* Jupyter Notebook

## Machine Learning Concepts

* Reinforcement Learning
* Multi-Armed Bandits
* Online Learning
* Bayesian Decision Making
* Exploration-Exploitation
* Statistical Evaluation

---

# 📦 Installation

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/movielens-multi-armed-bandit.git
cd movielens-multi-armed-bandit
```

---

## Create Virtual Environment

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
jupyter
```

---

# ☁️ Running on Google Colab

The notebook is designed to work with Google Colab.

Upload:

```text
ml-1m.zip
```

to:

```text
/content/ml-1m.zip
```

The notebook automatically:

1. Detects the ZIP file.
2. Extracts it.
3. Loads the `.dat` files.
4. Processes the ratings.
5. Creates the bandit environment.
6. Runs all algorithms.
7. Generates evaluation plots.

The expected path is:

```text
/content/ml-1m.zip
```

---

# 💻 Running Locally

Download the MovieLens 1M dataset and place it in the project directory.

Example:

```text
project/
│
├── data/
│   └── ml-1m.zip
│
└── notebooks/
    └── movielens_multi_armed_bandit.ipynb
```

Modify the dataset path in the notebook:

```python
DATASET_PATH = "../data/ml-1m.zip"
```

Then start Jupyter:

```bash
jupyter notebook
```

---

# 🔄 Implementation Workflow

The notebook follows the following process.

## Step 1 — Load Dataset

```text
ml-1m.zip
     ↓
Extract
     ↓
ratings.dat
movies.dat
users.dat
```

---

## Step 2 — Load Ratings

```text
UserID
MovieID
Rating
Timestamp
```

---

## Step 3 — Convert Ratings to Rewards

```text
Rating >= 4 → 1
Rating < 4  → 0
```

---

## Step 4 — Calculate Movie Statistics

For each movie:

```text
Total Ratings
Average Rating
Positive Ratings
Positive Rate
```

---

## Step 5 — Select Arms

The top 100 most-rated movies are used as the initial arms.

---

## Step 6 — Create Environment

Each movie receives a historical positive-rating probability.

---

## Step 7 — Initialize Algorithm

One of the following:

```text
Random
Epsilon-Greedy
Decaying Epsilon-Greedy
UCB
Thompson Sampling
```

---

## Step 8 — Select Movie

The algorithm selects an arm.

---

## Step 9 — Generate Reward

The simulated environment generates:

```text
Reward = 0
```

or:

```text
Reward = 1
```

---

## Step 10 — Update Algorithm

The algorithm updates its estimates.

---

## Step 11 — Repeat

This process is repeated thousands of times.

---

## Step 12 — Evaluate

The algorithm is evaluated using:

```text
Total Reward
Average Reward
Cumulative Reward
Cumulative Regret
Movie Selection Frequency
```

---

# 📊 Evaluation Metrics

## 1. Total Reward

The total reward is:

[
R_T =
\sum_{t=1}^{T}R_t
]

Higher is better.

---

# 2. Average Reward

[
\bar{R}
=======

\frac{1}{T}
\sum_{t=1}^{T}R_t
]

Higher average reward indicates that the algorithm is making better decisions on average.

---

# 3. Cumulative Reward

Cumulative reward at timestep (t):

[
C_t =
\sum_{i=1}^{t}R_i
]

A good algorithm should show a steep cumulative reward curve.

---

# 4. Regret

Regret measures how much reward was lost compared to always choosing the optimal arm.

For selected arm (a_t):

[
r_t =
\mu^*-\mu_{a_t}
]

where:

* (\mu^*) = expected reward of the optimal movie
* (\mu_{a_t}) = expected reward of selected movie

Cumulative regret:

[
R_T =
\sum_{t=1}^{T}
(\mu^*-\mu_{a_t})
]

Lower regret is better.

---

# 5. Movie Selection Frequency

The notebook also records how often each movie is selected.

A successful algorithm should increasingly select high-performing movies.

Example:

```text
Thompson Sampling

Movie A → 50 selections
Movie B → 120 selections
Movie C → 7200 selections
Movie D → 80 selections
...
```

---

# 🧪 Experiments

The notebook performs several experiments.

---

## Experiment 1 — Dataset Exploration

The notebook analyzes:

* Number of users
* Number of movies
* Number of ratings
* Rating distribution
* Positive reward distribution
* Movie popularity

---

## Experiment 2 — Epsilon-Greedy

Configuration:

```text
epsilon = 0.1
```

The algorithm explores approximately 10% of the time.

---

## Experiment 3 — Decaying Epsilon-Greedy

Configuration:

```text
Initial epsilon = 1.0
Minimum epsilon = 0.01
Decay = 0.999
```

The algorithm starts with substantial exploration and gradually exploits learned information.

---

## Experiment 4 — UCB

Configuration:

```text
c = 2.0
```

The algorithm uses an uncertainty-based exploration bonus.

---

## Experiment 5 — Thompson Sampling

Uses:

```text
Beta(1, 1)
```

as the initial prior for every movie.

---

## Experiment 6 — Algorithm Comparison

All algorithms are compared using:

```text
Cumulative Reward
Cumulative Regret
Average Reward
```

---

## Experiment 7 — Epsilon Sensitivity

Different values of epsilon are evaluated:

```text
0.01
0.05
0.10
0.20
0.30
```

This demonstrates how the exploration rate affects performance.

---

## Experiment 8 — Multiple Independent Runs

Because the environment is stochastic, a single experiment can produce noisy results.

The notebook therefore runs multiple independent experiments.

Example:

```text
20 runs
×
10,000 steps
```

The results are averaged to obtain a more reliable comparison.

---

# 📈 Results

The notebook produces several visualizations.

## Cumulative Reward

```text
Algorithm
   │
   ├── Random
   ├── Epsilon-Greedy
   ├── Decaying Epsilon-Greedy
   ├── UCB
   └── Thompson Sampling
            │
            ▼
      Cumulative Reward
```

A better algorithm should generally accumulate reward faster.

---

## Cumulative Regret

The regret plot measures the cost of exploration.

A better algorithm should generally have a lower regret curve.

---

## Learning Curves

The running average reward shows how quickly an algorithm approaches the reward level of good arms.

---

## Selection Frequency

The notebook shows which movies each algorithm selects most frequently.

This helps visualize whether an algorithm successfully identifies high-reward movies.

---

# 🧠 Expected Behavior

Suppose the simulator identifies:

```text
Movie A → 0.40
Movie B → 0.55
Movie C → 0.75
Movie D → 0.35
Movie E → 0.60
```

The optimal movie is:

```text
Movie C
```

At the beginning:

```text
Algorithm
   ↓
Explore multiple movies
```

After learning:

```text
Movie C
   ↓
High selection frequency
```

The exact results will vary because rewards are randomly sampled.

---

# 📌 Important Offline Evaluation Note

This is one of the most important considerations when interpreting this project.

MovieLens is a **static historical dataset**.

A real online bandit system would work like:

```text
Recommend Movie
       ↓
Real User
       ↓
User Response
       ↓
Reward
       ↓
Update Bandit
```

The current project instead does:

```text
Historical MovieLens Data
       ↓
Estimate Historical Reward Probability
       ↓
Simulated Environment
       ↓
Generate Reward
       ↓
Update Bandit
```

Therefore, this project demonstrates the **algorithmic behavior of Multi-Armed Bandits**, but it is not an actual online recommendation deployment.

This distinction is important because offline recommendation datasets do not automatically provide counterfactual information about what would have happened if a different movie had been recommended.

---

# ⚠️ Limitations

## 1. No User Context

The basic implementation treats each movie as an arm without considering the individual user.

Therefore:

```text
User A → Movie X
User B → Movie X
```

are effectively treated the same way.

A real recommendation system should account for user preferences.

---

## 2. Global Movie Popularity

The current bandit primarily learns which movies have high historical positive-rating probabilities.

It does not learn:

```text
Which movie is best for THIS user?
```

It learns approximately:

```text
Which movie is good for users in general?
```

---

## 3. Simulated Environment

The rewards are generated using historical probabilities.

There is no actual user interaction during the experiment.

---

## 4. Stationary Rewards

The environment assumes that movie reward probabilities do not change.

Real user preferences can change over time.

---

## 5. Binary Reward

The original 1–5 rating is converted into:

```text
0 / 1
```

This loses some information.

A 3-star rating and a 1-star rating both become:

```text
Reward = 0
```

Similarly:

```text
4-star and 5-star
```

both become:

```text
Reward = 1
```

---

## 6. Offline Data Bias

The MovieLens dataset contains historical interactions that were not generated by the bandit policy.

This creates an important offline evaluation challenge.

---

# 🚀 Future Improvements

Several extensions can make the project substantially more realistic.

## 1. Contextual Bandits

Use user features:

```text
Age
Gender
Occupation
Previous Ratings
Movie Genres
```

and select movies based on the current user.

---

## 2. LinUCB

Implement:

[
LinUCB
]

which incorporates feature vectors into the UCB framework.

---

## 3. Contextual Thompson Sampling

Extend Thompson Sampling to incorporate user and movie features.

---

## 4. User-Specific Recommendations

Instead of one global bandit:

```text
One Bandit
   ↓
All Users
```

use:

```text
User 1 → Bandit
User 2 → Bandit
User 3 → Bandit
...
```

---

## 5. User-Movie Context

Represent an interaction using:

```text
User Features
      +
Movie Features
      ↓
Context Vector
```

Then predict the expected reward for the user-movie pair.

---

## 6. Non-Stationary Bandits

Allow reward distributions to change over time.

For example:

```text
Time 1:
Movie A performs best

Time 2:
Movie B performs best

Time 3:
Movie C performs best
```

This would make the environment more realistic.

---

## 7. Continuous Rewards

Instead of:

```text
Rating >= 4 → 1
Rating < 4 → 0
```

use normalized ratings:

[
Reward =
\frac{Rating-1}{4}
]

which produces:

```text
1 star → 0.00
2 stars → 0.25
3 stars → 0.50
4 stars → 0.75
5 stars → 1.00
```

---

## 8. Real-Time Recommendation API

A future version could expose the bandit through FastAPI:

```text
React Frontend
      ↓
FastAPI
      ↓
Bandit Model
      ↓
Movie Recommendation
      ↓
User Feedback
      ↓
Bandit Update
```

---

## 9. Interactive Dashboard

A Streamlit dashboard could display:

* Current best movie
* Arm selection distribution
* Reward history
* Regret
* Algorithm comparison
* User recommendations

---

# 🧩 Contextual Bandit Extension

The current project can be extended into a Contextual Bandit system.

Instead of:

```text
Movie → Reward
```

we use:

```text
User + Movie → Reward
```

For example:

```text
User Features
│
├── Age
├── Gender
└── Occupation
        │
        ▼
   Context Vector
        │
        +
        │
Movie Features
│
├── Genre
└── Historical Performance
        │
        ▼
Contextual Bandit
        │
        ▼
Recommended Movie
        │
        ▼
User Feedback
        │
        ▼
Reward
```

This would transform the project into a personalized recommendation system.

---

# 🌎 Real-World Applications

Multi-Armed Bandits are useful in many systems where actions must be chosen while learning from feedback.

## Advertisement Optimization

```text
User
 ↓
Choose Ad
 ↓
Click / No Click
 ↓
Reward
```

---

## Recommendation Systems

```text
User
 ↓
Choose Movie/Product/Article
 ↓
User Engagement
 ↓
Reward
```

---

## A/B Testing

Instead of permanently assigning users equally to variants, a bandit can gradually allocate more traffic to better-performing variants.

---

## Content Recommendation

Bandits can optimize:

* News articles
* Videos
* Posts
* Notifications
* Headlines
* Images

---

## Online Personalization

Bandits can learn which actions work best for different users or contexts.

---

# 🔬 MAB vs Traditional Recommendation Systems

Traditional recommendation:

```text
Historical Data
      ↓
Train Model
      ↓
Predict Preference
      ↓
Recommend
```

Multi-Armed Bandit:

```text
Select Recommendation
      ↓
Observe Reward
      ↓
Update
      ↓
Select Again
```

The major advantage of a bandit is that it explicitly handles the **exploration-exploitation problem**.

---

# 🤖 MAB vs DQN

Multi-Armed Bandits are related to Reinforcement Learning but are simpler than algorithms such as DQN.

| Feature            | Multi-Armed Bandit        | DQN       |
| ------------------ | ------------------------- | --------- |
| State              | Usually absent            | Present   |
| Actions            | Arms                      | Actions   |
| Immediate reward   | Yes                       | Yes       |
| Future reward      | Limited                   | Important |
| State transitions  | No meaningful transitions | Yes       |
| Neural network     | Usually unnecessary       | Required  |
| Complexity         | Lower                     | Higher    |
| Long-term planning | No                        | Yes       |

A bandit is appropriate when:

> The main problem is choosing among actions based on immediate feedback.

DQN becomes more appropriate when:

> Actions affect future states and long-term rewards.

---

# 📚 Learning Outcomes

After completing this project, you should understand:

* What Multi-Armed Bandits are.
* How bandits differ from traditional supervised learning.
* How to formulate recommendation as a bandit problem.
* How to transform historical ratings into rewards.
* How Bernoulli reward environments work.
* How Epsilon-Greedy works.
* How Decaying Epsilon-Greedy works.
* How UCB works.
* How Thompson Sampling works.
* How Bayesian uncertainty can drive exploration.
* How to calculate cumulative reward.
* How to calculate regret.
* How to evaluate stochastic algorithms.
* Why multiple runs are important.
* Why offline bandit evaluation is difficult.
* Why Contextual Bandits are more appropriate for personalization.
* How a basic bandit can evolve into a recommendation system.

---

# 🧪 Reproducibility

The notebook sets a NumPy random seed:

```python
np.random.seed(42)
```

This helps make experiments more reproducible.

However, because the environment and algorithms are stochastic, results may still differ if:

* Random seeds are changed.
* The number of iterations is changed.
* The number of arms is changed.
* Hyperparameters are changed.

---

# ⚙️ Configurable Parameters

The main parameters can be changed directly in the notebook.

## Number of arms

```python
N_ARMS = 100
```

## Number of steps

```python
N_STEPS = 10000
```

## Number of experimental runs

```python
N_RUNS = 20
```

## Epsilon

```python
epsilon = 0.1
```

## UCB exploration coefficient

```python
c = 2.0
```

## Epsilon decay

```python
decay = 0.999
```

These parameters can be tuned to study their effect on performance.

---

# 📊 Example Experimental Questions

After running the notebook, several research questions can be investigated:

### Question 1

Which algorithm achieves the highest cumulative reward?

### Question 2

Which algorithm has the lowest cumulative regret?

### Question 3

How does epsilon affect Epsilon-Greedy?

### Question 4

How quickly does Thompson Sampling identify high-reward movies?

### Question 5

Does UCB explore unpopular movies more than Epsilon-Greedy?

### Question 6

How does the number of arms affect learning speed?

### Question 7

How does increasing the number of timesteps affect regret?

### Question 8

What happens when the reward threshold changes from:

```text
Rating >= 4
```

to:

```text
Rating >= 3
```

### Question 9

What happens when the environment becomes non-stationary?

These experiments can be used to extend the project beyond the basic implementation.

---

# 🔧 Possible Reward Experiments

The default reward definition is:

```text
Rating >= 4 → 1
Rating < 4  → 0
```

You can experiment with alternative definitions.

## Strict Positive Reward

```text
Rating == 5 → 1
Everything else → 0
```

## Moderate Positive Reward

```text
Rating >= 3 → 1
Rating < 3 → 0
```

## Continuous Reward

```text
1 → 0.00
2 → 0.25
3 → 0.50
4 → 0.75
5 → 1.00
```

Different reward definitions may lead to different algorithm behavior.

---

# 📌 Key Takeaways

The major concepts demonstrated by this project are:

```text
                 Multi-Armed Bandit
                         │
             ┌───────────┴───────────┐
             │                       │
        Exploration             Exploitation
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  Learn From Reward
                         │
                         ▼
                 Maximize Reward
                         │
                         ▼
                 Minimize Regret
```

The project demonstrates that intelligent decision-making does not always require a traditional supervised-learning model.

A system can learn directly from reward feedback.

---

# 🏁 Conclusion

This project applies classical Multi-Armed Bandit algorithms to the MovieLens 1M dataset and demonstrates how historical recommendation data can be transformed into a simulated online decision-making environment.

The project progresses through:

```text
MovieLens 1M
     ↓
Data Exploration
     ↓
Reward Engineering
     ↓
Movie → Bandit Arm
     ↓
Simulated Environment
     ↓
Random Baseline
     ↓
Epsilon-Greedy
     ↓
Decaying Epsilon-Greedy
     ↓
UCB
     ↓
Thompson Sampling
     ↓
Reward Analysis
     ↓
Regret Analysis
     ↓
Multiple Experiments
     ↓
Algorithm Comparison
```

The implementation provides a strong foundation for more advanced work in:

* Reinforcement Learning
* Online Learning
* Recommendation Systems
* Contextual Bandits
* Bayesian Decision Making
* Personalized Recommendation

The natural next step is to extend the project from a **global movie-selection bandit** to a **Contextual Bandit**, where user characteristics and movie characteristics are incorporated into the decision process.

---

# 👨‍💻 Author

**Aditya Udai**

B.Tech Computer Science and Engineering — Artificial Intelligence

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Reinforcement Learning
* Deep Learning
* Recommendation Systems
* Software Engineering
* Backend Development
* Full-Stack Development

---

# 📜 License

This project is intended for educational and research purposes.

Please refer to the MovieLens dataset's own terms and licensing requirements when using or redistributing the dataset.

The source code in this repository can be modified and extended for learning and experimentation.

---

# ⭐ If You Found This Project Useful

If this project helped you understand Multi-Armed Bandits, consider:

* ⭐ Starring the repository
* 🍴 Forking the project
* 🐛 Reporting issues
* 💡 Suggesting improvements
* 🚀 Extending it with Contextual Bandits

---

## Future Project Version

The next version of this project can extend the current architecture:

```text
                 MovieLens 1M
                      │
                      ▼
              User + Movie Features
                      │
                      ▼
             Contextual Bandit
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       LinUCB      Thompson     Neural
                  Sampling      Bandit
          │           │           │
          └───────────┼───────────┘
                      ▼
              Personalized Movie
                Recommendation
                      │
                      ▼
                  Reward
                      │
                      ▼
                Online Update
```

This would transform the current educational MAB implementation into a much more realistic **personalized recommendation and online learning system**.
