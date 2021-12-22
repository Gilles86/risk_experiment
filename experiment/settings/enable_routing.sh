ip link add name br0 type bridge
ip link set br0 up

ip link set eth0 up
ip link set eth0 master br0

ip addr del 192.168.1.3/24 dev eth0
ip addr add 192.168.1.3/24 dev br0

route add -net default gw 192.168.1.2
ip link set wlan0 master br0
