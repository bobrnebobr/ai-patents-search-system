FROM ubuntu:latest
LABEL authors="stupichev"

ENTRYPOINT ["top", "-b"]