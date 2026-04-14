#!/bin/bash

# Source the get_ip.sh script to import the IP_ADDRESS variable
source ./get_ip.sh

# Use the IP_ADDRESS variable
echo "The system's IP address is: $IP_ADDRESS"


# Conditionally set the IP_ADDRESS based on the INSTITUTE value
if [ "$IP_ADDRESS" == "192.168.12.1" ]; then
    export INSTITUTE="ABESIT"
elif [ "$IP_ADDRESS" == "192.168.43.101" ]; then
    export INSTITUTE="GLBITM"
else
    export INSTITUTE="ABESIT"
fi


# Update the master.env file (optional, for completeness)
sed -i "/^IP_ADDRESS=/c\IP_ADDRESS=$IP_ADDRESS" master.env
sed -i "/^INSTITUTE=/c\INSTITUTE=$INSTITUTE" master.env

# Update the secondary.env file
sed -i "/^IP_ADDRESS=/c\IP_ADDRESS=$IP_ADDRESS" STANDALONE-SERVICES/standalone-services.env
sed -i "/^INSTITUTE=/c\INSTITUTE=$INSTITUTE" STANDALONE-SERVICES/standalone-services.env
