pipeline {
    agent {
        docker {
            image 'mrsnoo123/jenkins-agent-aws-dvc:latest'
            args '-u root:root' // Run as root user to avoid permission issues
        }
    }

    environment {
        AWS_DEFAULT_REGION = 'us-west-2'
    }

    stages {
        stage('Checkout Repository') {
            steps {
                withCredentials([string(credentialsId: 'github-token', variable: 'GITHUB_TOKEN')]) {
                    git url: 'https://github.com/QuangNg14/MLOps-CIFAR10', branch: 'main', credentialsId: 'github-token'
                }
            }
        }

        stage('Set up Python') {
            steps {
                sh '''
                    python3 -m venv venv  # Create virtual environment
                    . venv/bin/activate   # Activate virtual environment
                    python3 -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Verify AWS Credentials') {
            steps {
                withCredentials([
                    string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh '''
                        echo "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
                        echo "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
                        aws s3 ls
                    '''
                }
            }
        }

        stage('Run Training') {
            steps {
                withCredentials([
                    string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh '''
                        . venv/bin/activate  # Activate virtual environment
                        export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
                        export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
                        export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
                        python3 train.py
                    '''
                }
            }
        }

        // stage('Run Flask Application') {
        //     steps {
        //         withCredentials([
        //             string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
        //             string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
        //         ]) {
        //             sh '''
        //                 . venv/bin/activate  # Activate virtual environment
        //                 export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
        //                 export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
        //                 export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
        //                 python3 train.py &
        //                 sleep 10 # Give the server some time to start
        //             '''
        //         }
        //     }
        // }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo 'Training and Flask application setup completed successfully!'
        }
        failure {
            echo 'Training or Flask application setup failed. Please check the logs.'
        }
    }
}
