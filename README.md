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
