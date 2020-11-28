# voiceinterfaceprogramming

docker build -t myimage . && docker run -p 5000:5000 myimage

curl -X POST -d "Hey" localhost:5000/detectIntent