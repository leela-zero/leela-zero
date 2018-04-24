# Run Leela Zero client on a Tesla K80 GPU for free (Google Colaboratory)

[Google Colaboratory](https://colab.research.google.com) (Colab) is a free tool for machine learning research. It's a Python notebook environment on a personal VM with a NVIDIA Tesla K80 GPU. Using Colab requires **no installation and runs in your browser**.

This example shows how to run a **Leela Zero client on the K80 GPU to contribute training games**. You can expect to contribute 60-80 games/day.

Google offers **free and unlimited access to the GPU**, but each session will **stop running after 12 hours of use and need to be restarted**. You must also keep the browser tab open. More details are below.

**Do not use multiple accounts for training.** Google has notified us they will block users for this.

## Running the GPU client

* Sign in to your Google account and [open the notebook in Google Colab](https://colab.research.google.com/drive/1WQfPOFhkahIJSxdIjeSQqK4q30j3T1qF).
* **File** -> **Save a copy in Drive…**.
* When the copied notebook opens, click **Runtime** -> **Run All**, which will run each of the cells in order. This will take around 10 minutes to complete.

Note: Google offers **unlimited access to the GPU**, but each session will **stop running after 12 hours of use and need to be restarted**. The animated spinning "stop" symbol will turn into a static red "play" symbol when the cell has stopped. You can restart with **Runtime** -> **Restart Runtime** followed by **Runtime** -> **Run All**. A simple macro would work to automate the restarting process. 

A session will also stop if you close the browser tab running Colab (about ~1.5 hours after closing the tab). To ensure the client runs for the full 12 hours, please **keep the tab open**.

## Troubleshooting
 * If you get a **"signal: aborted (core dumped)" error** when running the client or **"failed to assign a backend"** popup (examples below), there are no GPUs available on Google Colab. Try **Runtime** -> **Restart Runtime** and running again, or **kill the entire VM** with `!kill -9 -1` and try again (VM may take 5 minutes to restart after being killed). **As Google Colab has a limited number of free GPUs, you may just have to try again another time.**
```
cl::Error
what():  clGetPlatformIDs
2018/04/18 14:52:31 signal: aborted (core dumped)
```
![No GPUs](https://i.imgur.com/UI63IrA.png)
 * If the notebook appears to be stuck in "Initializing" and won't run, try restarting as above. After restart, you should see "Connected" with a green checkmark.

## Other Platforms
 * Other paid platforms offer a similar service as Google Colab (Jupyter notebook Python environment for machine learning). For example, [FloydHub](https://www.floydhub.com/) offers a free 2-hour Tesla K80 GPU trial, and a [working Jupyter notebook is available here](https://drive.google.com/open?id=1c0rxfB5r-5-JhfNAjJfvjDFBSVYIFOq7) (developed by @scs-ben).
