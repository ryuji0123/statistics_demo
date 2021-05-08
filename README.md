# Statistics demo to understand definitions and theorems
## Installation Steps
1. clone
```sh
$ git clone git@github.com:ryuji0123/statistics_demo.git
```

2. environment setup

The names of the docker image and container are specified by constants described in docker/env.sh.
These constants can be edited to suit your project.
```sh
$ cd statistics_demo
$ cp docker/.env.sh docker/env.sh
$ sh docker/build.sh
$ sh docker/run.sh
$ sh docker/exec.sh
```

## Development Steps
1. add your demo
2. run flake8 scripts
```sh
$ sh build.sh
```

## Visualize distributions
In this section, you can visualize and save distirbution figures
#### Supported distributions
|Distribution type|Function type|
|:---|:---|
|Gaussian|Probability Density Function|
|Gaussian|Cumulative Distribution Function|
|Beta|Probability Density Function|
|Beta|Cumulative Distribution Function|

The figures will be saved in figs/.
- Run this demo. You can use the command:
```
$ python src/distribution_function.py
```


#### Case 1. Gaussian

<img src=https://user-images.githubusercontent.com/49121951/116657667-c98e5100-a9c9-11eb-9651-891a6ae53ee5.png width='470px'><img src=https://user-images.githubusercontent.com/49121951/116657716-d9a63080-a9c9-11eb-9671-2455ed4f6dd9.png width='450px'>

#### Case 2. Beta
<img src=https://user-images.githubusercontent.com/49121951/116657825-05291b00-a9ca-11eb-9172-18162e4ced07.png width='470px'><img src=https://user-images.githubusercontent.com/49121951/116657853-14a86400-a9ca-11eb-86e1-d8f4077333a8.png width='450px'>
