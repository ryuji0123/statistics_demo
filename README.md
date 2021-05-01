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

## Demo
### gauss_pdf_and_cdf.py

Show and save some distribution figures.

The figures supported in this demo are shown at the follow table.
|distribution type|function type|
|:---|:---|
|gaussian|PDF|
|gaussian|CDF|
|beta|PDF|
|beta|CDF|

PDF is probability distribution function and CDF is cumulative distribution function.

- Run this demo. You can use command:
```
$ python src/gauss_pdf_and_cdf.py
```

If you specify "gaussian" as the argument of the main function, show gaussian PDF and CDF,
![gaussian_pdf](https://user-images.githubusercontent.com/49121951/116657667-c98e5100-a9c9-11eb-9651-891a6ae53ee5.png)
![gaussian_cdf](https://user-images.githubusercontent.com/49121951/116657716-d9a63080-a9c9-11eb-9671-2455ed4f6dd9.png)

If you specify "beta" as the argument of the main function, show beta PDF and CDF,
![beta_pdf](https://user-images.githubusercontent.com/49121951/116657825-05291b00-a9ca-11eb-9172-18162e4ced07.png)
![beta_cdf](https://user-images.githubusercontent.com/49121951/116657853-14a86400-a9ca-11eb-86e1-d8f4077333a8.png)

The figures are saved in figs/.
