<h3 align="center"><b>APS FAULT DETECTION</b></h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> The <b>Air Pressure System (APS)</b> is a critical component of a heavy-duty vehicle that uses compressed air to force a piston to provide pressure to the brake pads, slowing the vehicle down. The benefits of using an APS instead of a hydraulic system are the easy availability and long-term sustainability of natural air. This project pivots to solve the binary classification problem, in which it is to be determined whether the fault in a given vehicle is due to APS or not, aiming to keep the false negatives as low as they could be.<br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Built Using](#built_using)
- [Deployment](#deployment)


## 🧐 About <a name = "about"></a>

### <b>Problem Statement</b>

- The system in focus is the Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes. This is a Binary Classsification problem, in which the **affirmative class corresponds to component failures for a specific component of the APS system** and apparently **the negative one corresponds to trucks with failures for components not related to the APS system**.

- Cost of dealing with an actual APS component failure (or any other failure) is presumed as 10, however the same cost is taken as 500 when a faulty truck is missed due to whatever reasons (as it might cause a horrible breakdown of the vehicle).

- The total cost to be predicted by the model is the sum of `Cost_1` multiplied by the number of instances with type 1 failure and `Cost_2` with the number of instances with type 2 failure, resulting in a `Total_cost`. In this case **`Cost_1` refers to the cost that an unnessecary check needs to be done by a mechanic at the workshop**, while **`Cost_2` refers to the cost of missing a faulty truck, which may cause a breakdown**.

- `Total_cost = Cost_1 * No_Instances + Cost_2 * No_Instances.`

### <b>Solution Proposed</b>

Two **Machine Learning pipelines** are employed in this project -- one for learning pertinent sensors' relations to faults from the past data and the other for making predictions on the new ones. 

From the looks of the problem statement it's very evident that cost due to unnecessary repairs has to be brought down to crest and as such the false predictions. More importantly, emphasis is to be placed on **reducing the false negatives, as cost incurred due to them -- missing a faulty vehicle due to whatever reasons -- is 50 times higher than the false positives** and also because if missed even by mistake, then the vehicle may have severe breakdown leading to much bigger problems.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### <b>Prerequisites</b>

This project requires the [Python](https://www.python.org/downloads/), [Pip-PyPI](https://pip.pypa.io/en/stable/installation/) and [Docker](https://www.docker.com/) installed. Apart from these, you do need to have an [AWS](https://aws.amazon.com/?nc2=h_lg) account to access the services like [Amazon ECR](https://aws.amazon.com/ecr/), [Amazon EC2](https://aws.amazon.com/ec2/?nc2=type_a), and [Amazon S3](https://aws.amazon.com/s3/).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included.

**NOTE:** All the other dependencies/applications/tools will be installed when you'd get your virtual env ready and install the requirements present in the **requirements.txt** file.

### <b>Installing</b>

A step by step series of examples that tell you how to get the development env running.

**i.** First and foremost, create a virtual environment,

```
conda create -n hey_sensor python==3.7
```

..accompanied by the activation of the created environment.

```
conda activate hey_sensor
```

**ii.** Now, install the requirements for this project.

```
pip install -r requirements.txt
```

**iii.** Setup a database, now, in [MongoDB Atlas](https://www.mongodb.com/atlas/database) and copy the connection string from there, followed by creating a .env file in the development env. Then, create an env variable `MONGO_DB_URL` and assign the connection string to it.

```
MONGO_DB_URL = <connection_string>
```

**iv.** Now to orchestrate and monitor the **machine learning pipelines/workflows**, run 

```
# The Standalone command will initialise the database, make a user,
# and start all components for you.

airflow standalone
```
..followed by visiting `localhost:8080` in the browser. Can now use the admin account details shown on the terminal to login. 

**v.** At the moment, you're good to run the training and prediction pipelines from the airflow UI as per the requirement.


## ⛏️ Built Using <a name = "built_using"></a>

- [MongoDB](https://www.mongodb.com/) - Database
- [Airflow](https://airflow.apache.org/) - Scheduler and Orchestrator of Pipelines and Workflows
- [Docker](https://www.docker.com/) - To Containerize the Application
- [Github Actions](https://github.com/features/actions) - For Continous Integration and Continous Delivery


## 🚀 Deployment <a name = "deployment"></a>

To deploy this application on a live system, we are gonna use the **AWS** cloud platform. The flow would go like this -- A docker image in regard to instructions specified in `Dockerfile` from GitHub is going to be pushed to [Amazon ECR (Elastic Container Registry)](https://aws.amazon.com/ecr/), from where it's gonna be pulled to [Amazon EC2 (Elastic Compute Cloud)](https://aws.amazon.com/ec2/?nc2=type_a) wherein it'll run and build the apparent docker container. **All these steps are automated via [GitHub Actions](https://github.com/features/actions).**

Whence the training pipeline concludes in the airflow UI, all the artifacts and models built in the process are gonna be synced to [Amazon S3](https://aws.amazon.com/s3/) bucket as instructed in the **training airflow dag**.

```
# training airflow dag

training_pipeline >> sync_data_to_s3
```

The batches on which predictions are to be made can be uploaded directly to the [Amazon S3](https://aws.amazon.com/s3/) bucket from where they'll be downloaded to the airflow and the prediction pipeline will run atop of them returning an output file for each uploaded batch containing the status of all the sensors, and as such those output prediction files will be synced to the prediction dir in **Amazon S3**, in regard to the flow defined in the **prediction airflow dag**.

```
# prediction airflow dag

download_input_files >> generate_prediction_files >> upload_prediction_files
```


Now, follow the following series of steps to deploy this application on a live system:

**NOTE:** Following steps are to be followed only after you commit and push all the code into a maintainable GitHub repository.

<br>

**i.** First off all, login in to the AWS using your AWS account credentials. Then, go to the **IAM** section and create a user `<username>` there, followed by selection of `Access key - Programmactic Access` for the **AWS access type**. Going next you'll find the **Attach existing policies directly** tab, where you have to check `Admininstrator Access` option. 

Now, upon creation of the user, you'll see an option to download csv file, from which the three necessary credentials oughta be gathered -- **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY** and **AWS_REGION**. As such, download the csv file and store these credentials somewhere.

<br>

**ii.** Now, create a repository in [Amazon ECR](https://aws.amazon.com/ecr/) where the docker image is to be stored and can be pulled into [Amazon EC2](https://aws.amazon.com/ec2/?nc2=type_a) and be put to use, as per the need/desire.

After this, copy the URI of the created repository from the ECR repositories section and assign it to the **AWS_ECR_LOGIN_URI** and **ECR_REPOSITORY_NAME** accordingly and store these variables somewhere secure.

```
# Say, for example
URI = 752513066493.dkr.ecr.ap-northeast-1.amazonaws.com/aps-fault-detector

# Assign accordingly
AWS_ECR_LOGIN_URI = 752513066493.dkr.ecr.ap-northeast-1.amazonaws.com
ECR_REPOSITORY_NAME = aps-fault-detector
```
<br>

**iii.** As of now, [Amazon S3](https://aws.amazon.com/s3/) bucket and [Amazon EC2](https://aws.amazon.com/ec2/?nc2=type_a) instance are yet to be created. Let's get on with the S3 bucket!

Go to the **Amazon S3** section, get started with creating a S3 bucket by choosing a name that is universally unique, followed by choosing the correct `AWS Region`. Upon creation of the said bucket, assign the bucket name to the variable **BUCKET_NAME** and store it somewhere secure.

<br>

**iv.** Similarly, let's get going with the EC2 instance. Go to the instances section of the EC2, and launch a new  EC2 instance escorted by choosing a name and a relevant `Application and OS Image`(say, **Ubuntu** for this project) for it. After that in the `Key pair (login)`, create a new key pair upon which a `.pem` file will be downloaded. Now, you can configure storage as per the requirement for this instance and finally get done with its launch.

At the moment, you have to wait for the `Status check` of launch of this instance to be passed.

<br>

**v.** Now, to access this instance with not only the SSH request but with the HTTP request too, we've gotta configure this setting. For this matter, go to the `Edit inbound rules` of this instance and add a new rule -- Make the **Type** as `All traffic` and the **Source** as `Anywhere-IPv4` -- and save this rule.

<br>

**vi.** Click on the `connect` button of the launched instance and the terminal will get open.

Now, in the terminal, execute the following commands, one at a time, to get the **Docker** up and running.

```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

<br>

**vii.** From the **runners** section in the **Actions settings** of your GitHub repo, create a new **linux** `self-hoster runner` and copy the commands from therein and execute them one by one in the EC2 instance terminal.

**Download**

```
# Create a folder
$ mkdir actions-runner && cd actions-runner# Download the latest runner package
$ curl -o actions-runner-linux-x64-2.301.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.301.1/actions-runner-linux-x64-2.301.1.tar.gz# Optional: Validate the hash
$ echo "3ee9c3b83de642f919912e0594ee2601835518827da785d034c1163f8efdf907  actions-runner-linux-x64-2.301.1.tar.gz" | shasum -a 256 -c# Extract the installer
$ tar xzf ./actions-runner-linux-x64-2.301.1.tar.gz
```

**Configure**

```
# Create the runner and start the configuration experience
$ ./config.sh --url https://github.com/suryanshyaknow/APS-fault-detection --token APNAENNPAOWALIWWN55VAELD2BO4A
# Last step, run it!
$ ./run.sh
```
**NOTE:** Upon the successful addition of the runner, when it's being asked on the terminal the name of the runner, enter `self-hosted`.

**Using your self-hosted runner**

```
# Use this YAML in your workflow file for each job
runs-on: self-hosted
```

<br>

**viii.** Now, at last, gather all those variabes that you were earlier asked to store securely -- **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**, **AWS_REGION**, **AWS_ECR_LOGIN_URI**, **ECR_REPOSITORY_NAME**, **BUCKET_NAME**, and at last but not the least, **MONGO_DB_URL** -- and add them as the **Secrets** in the `Actions Secrets and Variables` section of your GitHub repo. 

With this, the deployment setup has ultimately been configured. You can now access the deployment link of this application which is the `Public IPv4 DNS` address of the said EC2 instance.

---

