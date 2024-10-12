docker run -it \
    --name docker_dbo \
    --rm \
    --privileged \
    --net=host \
    -v "/Users/emanuelemengoli/epfl:/home/user/workspace" \
    ubuntu-wdbo

