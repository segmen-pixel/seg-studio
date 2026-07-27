# -- Build stage --
FROM node:20-alpine AS build
WORKDIR /app
COPY apps/trainer_ui/package.json /app/package.json
COPY apps/trainer_ui/package-lock.json /app/package-lock.json
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY apps/trainer_ui /app
RUN npm run build

# -- Serve stage --
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Route all non-file requests to index.html (SPA support) and proxy API
COPY apps/trainer_ui/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
