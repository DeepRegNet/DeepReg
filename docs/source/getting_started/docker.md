# Install Docker

https://docs.docker.com/get-docker/

For Linux based OS, there are some post steps. Otherwise you might have permission
denied errors later.

https://docs.docker.com/engine/install/linux-postinstall/

# Build image

```bash
docker build . -t deepreg -f Dockerfile
```

where

- `-t` names the built image as `deepreg`.
- `-f` provides the docker file for configuration.

# Run a container

`docker run --name <container_name> --privileged=true -ti <image_name> bash`

where

- `--name` names the container.
- `--privileged=true` is required to solve the permission issue linked to TensorFlow
  profiler.
- `-it` allows interaction with container and enters the container directly, more info
  on
  [stackoverflow](https://stackoverflow.com/questions/48368411/what-is-docker-run-it-flag.

# Others

- `docker rm -v <container_name>` removes a created container and its volumes, more info
  on [docker documentation](https://docs.docker.com/engine/reference/commandline/rm/).
