i dont know how to embbed images here, but you can find all pictures at :
https://imgur.com/a/AeoKUYa

# Quick facts before starting :

- This free trial is entirely and totally free of charge, without any obligatory end of free credit condition (this is the case for Google Cloud, i don't know about the other services)
- With powerful GPU like the Tesla V100, you will be able to produce 16 games per hour (all 5% resign) for leela-zero 40 blocksx256, which is much more than what a public card like a GTX 1080 Ti can do.
- To prevent abuse, spam, robots, etc, an id check will be performed with a valid credit card, but you will not be charged at all during the free trial
- These instructions will create an entirely automated leela-zero autogtp VM instance thanks to a startup-script in metadata : after setting it up correctly, it will not require any operation and will install all needed packages, compile and run leela-zero with autogtp, and will produce games automatically
- The instance uses cloud ressource, not your personal machine
- The instance is running on a server : it will stay online independently from you (even if your computer is powered off)
- This instance will be Preemptible : it uses cloud ressource that are not always available, causing it to be much cheaper but the instance is ephemere : after 24 hours max it cannot "live" anymore and will be terminated by preemptible use rules
- The Preemptiible terminations will not be a problem though, because our instance will be in a managed instance group 
- Our managed instance group will automatically create our first instance, install all needed packages on it, then automatically reboot it and automatically starting to produce games with autogtp
- Everytime our instance "dies" (max 24 hours because of preemptibility, or if you manually delete it), our managed instance group will automatically delete our "dead" instance and automatically recreate a new "child" preemtible instance (a new one, does not contain old data of the "parent" instance)
- Then, our managed instance group will automatically restart our new "child" instance, install all needed packages including leela-zero (takes exactly 10 minutes), then auto reboot, and then at reboot automatically start to produce games with autogtp, until the "child" instance dies, giving birth by the group to a new "child of the child" instance, etc.
- The exception to this automated recreation+autostart by the instance group is for scheduled maintainance by Google (rare, once every few weeks) which will require you to manual restart the instance (it takes 1 minute), then the auto-start script will handle everything again.




# Cloud Companies :

We are not affiliated with any cloud company, and we provide these instructions as they are a free of charge way to help public  contributing to leela-zero project.

We are thankful to these cloud companies for giving us these free trial opportunities.

The instructions below are for Google Cloud Free Trial as Google is a widespread company, but much of the documentation here can be used if slightly modified for other cloud companies offering similar cloud free trials that include a GPU (Microsft Azure, Oracle Cloud, etc.)


# Video Tutorial

As an interactive help to text instructions, a video tutorial is provided here :
youtube....(link will be added later)


# Start the Google Cloud Free Trial (id check)

You'll first need to start your google cloud free trial here :

https://console.cloud.google.com/

The cloud public ressource available being limitied, especially when it comes to powerful GPU like V100, in order to prevent abuse, spam, robots, multi accounts, etc, Google will ask to check your id with a valid credit card, but you will not be charged anything at all, even when your free trial credit ends.

And this free trial does not force you to susbcribe to anything at all. It is indeed entirely free of charge.

Also, just like for Google Colab, please do NOT try to use multiple accounts in any way, as Google won't hesitate to ban you, as it happened for google Colab users. If you want to help leela-zero, please rather try to spread these instructions so that more people join us in this contributed effort.

# Getting Started

If you are not redirected, after your id is successfully checked, you will be redirected here :

https://cloud.google.com/getting-started

To navigate in Google Cloud menus, click on the sandwich bar in the top left.

# Free credit consumption estimation : 

You can check your remaining free credit, in the billing menu of the sandwich bar.

Or, you can alternately use this link :

https://console.cloud.google.com/billing

At start, a free 300$ (or 257€) free trial credit is given to you, free of charge.
With the settings later explained, our instance will consume 0,774$/hour with a Tesla V100

This will allow us to use it for arround 390 hours, which would produce arround 6000 games leela-zero 40 blocks (5% resign only, as 0% resign games take much more time) entirely free of charge !
These 300$ can be used for arround 16,1 days for a h24 7/7 use of a Tesla V100, entirely free of charge !

# Create an instance template

In google cloud console, go to the sandwich bar on top left, and click on :
Compute Engine -> Instance templates
Alternatively, you can use this link :
https://console.cloud.google.com/compute/instanceTemplates/

An instance template is a recording of the settings we want every new instance to be automatically recreated with.
Click on blue button "Create instance template" :

Then, choose these settings :

As you can see in the 3 screenshots below :

img05
img1
img2
img3

- template name : any name you want
- on machine type, click "customize"
- 4 vcpu / 5,5 GB ram
- 1 GPU : Tesla V100
- untick "Extend memory"
- Boot Disk : click on "change" and choose Ubuntu 18.04 LTS with 10GB HDD (standard persistent disk)
- Firewall : allow http/https
Then click on "Management, security,disks, networking, sole tenacy" :
These new options will appear :
- Automation : in startup script, copy paste all the script below :
```
#!/bin/bash
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' glances|grep "install ok installed")
echo Checking for glanceslib: $PKG_OK
if [ "" == "$PKG_OK" ]; then
  echo "No glanceslib. Setting up glanceslib and all other leela-zero packages."
  sudo apt-get update && sudo apt-get -y upgrade && sudo apt-get -y dist-upgrade && sudo add-apt-repository -y ppa:graphics-drivers/ppa && sudo apt-get update && sudo apt-get -y install nvidia-driver-410 linux-headers-generic nvidia-opencl-dev && sudo apt-get -y install clinfo cmake git libboost-all-dev libopenblas-dev zlib1g-dev build-essential qtbase5-dev qttools5-dev qttools5-dev-tools libboost-dev libboost-program-options-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev qt5-default qt5-qmake curl && git clone https://github.com/gcp/leela-zero && cd leela-zero && git checkout next && git pull && git clone https://github.com/gcp/leela-zero && git submodule update --init --recursive && mkdir build && cd build && cmake .. && cmake --build . && cd ../autogtp && cp ../build/autogtp/autogtp . && cp ../build/leelaz . && cd ../.. && cp -r leela-zero /home/mygooglename/ && rm -r leela-zero && sudo apt-get -y install glances zip && sudo reboot
else 
  sudo apt-get clean && cd home/mygooglename/leela-zero/autogtp/ && ./autogtp -g 2
fi
```
- Availability Policy : Preemptibility ON

IMPORTANT NOTE : 
in the script : replace `mygooglename` (twice) with either your real gmail account name (without the @gmail.com), or a custom linux username (for privacy, etc.) that you will choose immediately and overwrite mygooglename with (example : trololhahaxd in the video tutorial).

note 2 : in `nvidia-driver-410` , replace `410` version number by whatever latest version number you find here :
https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa

note 3 : an instance template cannot be modified/updated, if you want for example to change nvidia-driver-410 in the script, change username, or modify anything in the instance template, you will have to create a new instance template, then delete your current instance group, and create a new instance group with your latest instance template.

note 4 : i chose ./autogtp -g 2 in the script as it is 25% faster from my extensive long time tests

note 5 : this script includes NEXT BRANCH as it is much faster and includes all new improvements, but if you want the MASTER BRANCH startup-script, please see : -----add----link----

note 6 : credits for the script go to (i modified it) : https://stackoverflow.com/a/10439058

After instance template finishes creating, you will get a screen like that

img4

# Check the regions and subregions that can provide a Tesla V100 :

This step will be fast, but important in order to create a group in the correct regions.
In another browser tab, if you're still in Compute engine (else use the top left sandiwich bar), on the left panel you will see "instance groups", click on it.
Alternatively, you can use this link : 
https://console.cloud.google.com/compute/instances

To know which regions and subregions have a Tesla V100, click on "create instance" (we will not actually create an instance, but just scroll through regions and subregions, then exit it)
Then click on "custom" machine.
And for every region, if you can add a GPU, add 1 GPU, and check if among these GPU a Tesla V100 is providable, as shown in the screenshot below : 

img12

Repeat this step for every subregion inside a region and similarly for every other regions : 
At the time of these instructions, these are the regions and subregions that can provide a Tesla V100 :


us-central1 (iowa) (has v100 in 3 subregions : a f b)

us-west1 (oregon) (has v100 in 2 subregions : b a)

europe-west4 (netherlands) (has v100 in 1 subregion : a)

asia-east1 (Taiwan) (has v100 in 1 subregion : c)


But please remember that these can change in the future for the better or the worse, this is why all the methodology is described here.


# Create a managed instance group :

This managed instance group will take care of automatically recreating a new instance every time preemptible use shuts it down (frequent), or if it is deleted by you (but NOT for scheduled maintainance which will require a manual restart of the instance, much more rare).

If you're still in Compute engine (else use the top left sandiwich bar), on the left panel you will see "instance groups", click on it.
Alternatively, you can use this link : https://console.cloud.google.com/compute/instanceGroups/

Click on "Create instance group"

Then, choose these settings :

As you can see in the screenshots below : 

img11
img21

- group name : any name you want
- location : multi-zone, then click on more
As explained earlier choose only one of the regions that can provide a Tesla V100 in at least one of its subregions (zones).
The instance group that we are creating will continuously be in charge of trying to recreate and restart the instance in any of the subregions (zones) that we will choose until it finds one where hardware is providable on the cloud.
This is why it is preferred to choose a region that can provide a Tesla V100 in many of its subregions.
In this example, i chose the region us-west1 (oregon)
Then click on "configure zones"
- zone (subregion) : tick only the subregions (zones) that can provide a Tesla V100
In this example i ticked subregions a and b, and unticked subregion c.
- instance template : choose the template that we created earlier.
This template will tell what the group what hardware to use, and also include a startup-script that will automatically install all packages needed, then install leela-zero, and then at every boot will start autogtp. It is entirely automated
- autoscaling : off
- number of instances : 1
As free trial accounts have a GPU quota of 1, we will not be able to create anymore simultanously running instances with a GPU.
- Health check : no

Click on blue button "create".
After some time, we get a screen like this :

img31

if you go back to "VM instances" (https://console.cloud.google.com/compute/instances) :

You can see that a new instance has been created by the managed instance group
This instance will automatically update, upgrade and install all system packages then compile leela-zero and reboot, and then automatically run autogtp.

img 41

# In case of system package corruption : Delete the instance

The first boot installation takes exactly 10 minutes.

But if, very unfortunately, your instance gets stopped by preemptible use in these first 10 minutes, there is a high probability of package corruption, so i suggest you either run the journal as explained below in # to see if instance is runing fine, or you delete the instance (no need to create a new one, the managed group instance will automatically take care of this).

If you want to delete an instance, it is in Compute engine-> VM instances, at the right of your instance click on the 3 dots, choose "Delete", and confirm the message, as shown below : 

delete1


# The last obligatory thing you need to know :

How to manually restart an instance after it has been stopped due to scheduled maintainance :

As explained earlier, Preemptible instances are much cheaper but can't last more than 24 hours, and can be stopped sooner (frequent).

This case is not at all a problem though, because, every time the instance is stopped by preemtpible use or deleted by yourself, the managed instance group will immediately and continuously automatically keep trying to recreate and restart our instance in any of the subregions(zones) we selected earlier that have a Tesla V100 available,, until it succeeds.
You don't have to do anything at all for that.


However, sometimes both preemptible and non premptible instances will get stopped by Google Cloud due to scheduled maintainance events.

This case is rare (maybe once every few weeks), but Preemptible instances can't be automatically restarted after that, unlike non Preemptible instances.
In that case, the user (you) will have to manually restart it.
To do that, go on "VM instances" page in Compute engine (https://console.cloud.google.com/compute/instances) :

If the instance is stopped, you will see a grey square (STOPPED) instead of the usual green circle (STARTED), as you can see here : 

stop1

When all VM instances are stopped, the free credit stops being consumed, so you don't have to worry about failing to restart a stopped instance, you can do it anytime later.
Also, you can willingly chose to stop contributing for some time and stop your instance, then whenever you want start contributing again.

To restart a stopped instance, click on the 3 dots at the right of your instance, and click on "start", as shown below :

start1
start2

note : starting the isntance will start the free trial credits consumption, as explained in the pre-start message

If you put your real gmail account name (without the @gmail.com), this concludes the obligatory steps !
You are now a Tesla V100 leela-zero contributor !
You can exit chrome, power off your computer, you don't have to do anything ! It is entirely automatic (except for scheduled maintainance events requiring a manual restart)


If you chose a custom linux username, there is one last step for you :


# Change linux username :

NOTE : again, do this step only if you chose a custom linux name instead of your default gmail account name
skip this step if you're okay with using your real gmail account name publicly.

If you chose to have custom name (for privacy, etc.), you need to do this step only once, and all new instances will have the name you chose.

In Compute engine -> "VM instances" (https://console.cloud.google.com/compute/instances)

If your instance has started (green circle), click on SSH button.

A new black window will open which is a linux terminal (ubuntu here).
If the SSH connection fails, click on retry button until succeeds (it will eventually succeed as long as your instance is on "started" (green circle status))

In this SSH window, on the top right click on the wheel settings, then change linux username, as shown below :

ssh name1
ssh name2
ssh name3

Since your ubuntu username is now different, for the startup-script to work, we will need to delete this instance as explained earlier in # In case of system package corruption : Delete the instance
A new instance will automatically be created and started by the instance group.

You are now a leela-zero Tesla V100 custom name contributor !
The following optionnal instructions may interest you


# Optionnal extra steps

## Optionnal : How to connect to your instance with the SSH button

In Compute engine -> "VM instances" (https://console.cloud.google.com/compute/instances)
If your instance has started (green circle), click on SSH button.

A new black window will open which is a linux terminal (ubuntu here).
If the SSH connection fails, click on retry button until succeeds (it will eventually succeed as long as your instance is on "started" (green circle status))

This command line will allow us to do many operations, and see what is going on (game production speed, ram/cpu usage, etc.)

## Optionnal : Ubuntu terminal navigation options in SSH window(s)

a) In this SSH window, on the top right click on the wheel settings.
Then on Copy Settings -> tick this : "Copy/Paste with Ctrl+Shift + C/V"

Now, just like in ubuntu terminal, you can paste into the terminal, or copy from the terminal to your computer, with :
ctrl+shift+c or ctrl+shift+v

b) you can open MANY SSH windows, not just one (for example we will see later glances+journal in 2 separate SSH windows)

c) And just like in ubuntu terminal, you can navigate previous entered commands with up/down arrows

d) to validate a command, press ENTER

e) To break a live mode refreshing command, in the SSH window do "ctrl+c"
this will be useful to stop the live refreshing journal, glances live stats, or any other live refreshing command


## Optionnal : Using the journal to see what commands are running in the background

In the SSH window, you dont see directly the start-up script, but it is running in the background.
Therefore, you should NOT manually run anything related to leela-zero

To see what the startup-script is running, (commands, autogtp games, etc.), do :

`sudo journalctl -u google-startup-scripts.service -b -e -f`

`-b` `-e` and/or `-f` can be removed to navigate freely in the journal file depending on what you want, see `sudo journalctl --help` for more details (for example `-r` can be useful)

For example, at the first boot we can see that the script starts to install all packages :

journal1

Then, at 2nd reboot, autogtp starts its tuning because it is the first run (but on next run game production will start directly) :

note : you don't see moves generated until one game ends, because all the moves are part of the same line in journal, so you will not see it until the line ends (aka. until the game ends).

journal2
journal3

After some time, you can get something like this :
note : as said earlier, no resign games are much slower to produce, so game production speed may vary depending on the number of no resign games you have

test2

## Optionnal : see hardware usage (cpu,gpu,ram etc.) with glances

The script provides the installation of glances hardware monitoring tool
In the SSH window, do : 

`glances`

test1

glances tells us many noticeable information :

for example, in this instance glances tells us that this instance has been running for almost 13 hours now,
and journal tells us 111 games have been produced in 767 minutes

to break the journal live mode refresh, do : ctrl+c in the SSH window

another useful command you can run to check gpu usage is :
`nvidia-smi`


## Optionnal : Why we don't run upgrade and dist-upgrade maintainance

Because the instance will live less than 24 hours.
The child instance will get newest packages at the time it's running.

## Optionnal : Manually check latest nvidia-driver version available

if for some reason you dont want to go to the ppa website, you can run this command instead :
`apt-cache search nvidia`

## Optionnal : Save all SGF produced and export them to download them on your personnal computer :

Steps described in this github comment, with pictures : https://github.com/gcp/leela-zero/issues/1943#issuecomment-430977929

I added this paragraph to answer @kwccoin on the github issue linked above.

It may also be useful for different yet similar purposes :

first,
you need to change the startup-script like this (add -k all sgf to autogtp) :

```
#!/bin/bash
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' glances|grep "install ok installed")
echo Checking for glanceslib: $PKG_OK
if [ "" == "$PKG_OK" ]; then
  echo "No glanceslib. Setting up glanceslib and all other leela-zero packages."
  sudo apt-get update && sudo apt-get -y upgrade && sudo apt-get -y dist-upgrade && sudo add-apt-repository -y ppa:graphics-drivers/ppa && sudo apt-get update && sudo apt-get -y install nvidia-driver-410 linux-headers-generic nvidia-opencl-dev && sudo apt-get -y install clinfo cmake git libboost-all-dev libopenblas-dev zlib1g-dev build-essential qtbase5-dev qttools5-dev qttools5-dev-tools libboost-dev libboost-program-options-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev qt5-default qt5-qmake curl && git clone https://github.com/gcp/leela-zero && cd leela-zero && git checkout next && git pull && git clone https://github.com/gcp/leela-zero && git submodule update --init --recursive && mkdir build && cd build && cmake .. && cmake --build . && cd ../autogtp && cp ../build/autogtp/autogtp . && cp ../build/leelaz . && cd ../.. && cp -r leela-zero /home/mygooglename/ && rm -r leela-zero && sudo apt-get -y install glances zip && sudo reboot
else 
  sudo apt-get clean && cd home/mygooglename/leela-zero/autogtp/ && ./autogtp -g 2 -k allsgf
fi
```

since instance templates are not modfiable, you need to create a new instance template
then delete current instance group
then create a new instance group with this new template


secondly,
then, click on SSH button in google cloud console again to open a 2nd command line window
in the 2nd SSH window, run these commands :

```
cd leela-zero/autogtp
ls
#replace v1b in all these commands by whatever name you like, always a different one for every new archive
zip -r -0 v1b.zip allsgf
curl --upload-file ./v1b.zip https://transfer.sh/v1b.zip
```

then you will get a download link as i did in my screenshot

download link for my example (ctrl+shift+c in ubuntu terminal) :
https://transfer.sh/6Lza1/v1b.zip

optionnal :
view sgf uploaded in the allsgf folder, sorted by date :


steps explained here, with pictures : https://github.com/gcp/leela-zero/issues/1943#issuecomment-431047043
run these commands, in autogtp folder :

```
cd allsgf
ls -t
#(to go back : cd ..)
```

read order : 
1st column top to bottom, then go to column 2 top to bottom, then column 3 etc

note that the sgf are also sorted by time in the zip archive :
(2 more sgf were generated since i did this screenshot)

alternatively, you can also have a log if you check the journal file, as explained earlier


## Extra : Manual leela-zero compilation instructions :

As i found unintuitive the github official leela-zero compiling instructions, you can go to my pastebin link referencing :

https://pastebin.com/552UN25c

This pastebin contains :

- how to compile and run leela-zero with autogtp NEXT BRANCH : in part 6e
- same but for MASTER BRANCH : in part 6f
- hardware needs calculation for leela-zero autogtp : in part 5e

Remember : as everything is automated with the stratup-script, you should NOT run autogtp manually, or it will run twice, causing it to be less efficient, hardware bottlenecked, unstable, and slower !
