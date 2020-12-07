#!/bin/bash

git clone https://git.busybox.net/busybox
cd busybox
make defconfig
make
cd ..

git clone https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux-stable.git
cd linux-stable
make defconfig
make -j$(nproc)
cd ..

cd busybox
make install

mkdir _install/lib
cp /lib/x86_64-linux-gnu/libm.so.6 /lib/x86_64-linux-gnu/libresolv.so.2 /lib/x86_64-linux-gnu/libc.so.6 _install/lib

mkdir _install/lib64
cp /lib64/ld-linux-x86-64.so.2 _install/lib64

(
cat <<EOF
dir /dev 755 0 0
nod /dev/tty0 644 0 0 c 4 0
nod /dev/tty1 644 0 0 c 4 1
nod /dev/tty2 644 0 0 c 4 2
nod /dev/tty3 644 0 0 c 4 3
nod /dev/tty4 644 0 0 c 4 4
slink /init bin/busybox 700 0 0
dir /proc 755 0 0
dir /sys 755 0 0
EOF

find _install -mindepth 1 -type d -printf "dir /%P %m 0 0\n"
find _install -type f -printf "file /%P %p %m 0 0\n"
find _install -type l -printf "slink /%P %l %m 0 0\n"
) > filelist

gen_init_cpio filelist | gzip > ramdisk

cd ..
qemu-system-x86_64 -kernel /boot/vmlinuz-3.2.0-4-amd64 -initrd ramdisk