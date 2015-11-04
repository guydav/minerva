#!/bin/sh

echo Running $1 workers

for i in `seq 1 $1`;
do
	rqworker &
done

