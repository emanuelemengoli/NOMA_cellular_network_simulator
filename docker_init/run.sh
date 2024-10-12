docker run -it \
    --name docker_dbo \
    --rm \
    --privileged \
    --net=host \
    -v "/Users/emanuelemengoli/github repos/NOMA_net/project:/home/user/workspace" \
    ubuntu-wdbo

