#!/bin/bash
# This container's shell lacks the host group (gid 109) that owns /dev/kfd.
# A fresh devuser process via sudo picks up the kfdhost group added with usermod.
exec sudo -n -u devuser env PATH="$PATH" HOME="$HOME" "$@"
