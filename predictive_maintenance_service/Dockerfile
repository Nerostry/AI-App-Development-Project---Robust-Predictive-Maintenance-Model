FROM node:20-alpine

WORKDIR /app

# Copy dependency definitions
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all source files into container
COPY . .

# Expose app port
EXPOSE 3000

# Start command
CMD ["npm", "start"]