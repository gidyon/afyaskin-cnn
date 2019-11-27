FROM node:12-alpine3.9

# Create app directory
WORKDIR /app

# Install app dependencies
# A wildcard is used to ensure both package.json AND package-lock.json are copied
# where available (npm@5+)
COPY package*.json ./

RUN apk update && \
   apk add ca-certificates && \
   update-ca-certificates && \
   rm -rf /var/cache/apk/* && \
   apk add python3

RUN npm install

# Bundle app source
COPY . .

EXPOSE 80
CMD [ "node", "server.js" ]