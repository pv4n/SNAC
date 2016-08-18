#!/bin/bash
a=0
for i in *.jpeg; do
  new=$(printf "frame%05d.jpeg" "$a") #04 pad to length of 4
  mv -- "$i" "$new"
  let a=a+1
done