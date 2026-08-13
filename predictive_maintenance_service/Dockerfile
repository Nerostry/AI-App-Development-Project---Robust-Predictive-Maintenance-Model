# Step 1: Use an official Node.js base image
FROM node:20-alpine

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy package files first to leverage Docker layer caching
COPY package*.json ./

# Step 4: Install dependencies
RUN npm install

# Step 5: Copy the rest of application code
COPY . .

# Step 6: Expose the port application listens on (e.g., 3000)
EXPOSE 3000

# Step 7: Define the command to start application
CMD ["npm", "start"]