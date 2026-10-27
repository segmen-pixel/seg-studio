# -- Build stage --
# The UI keeps the depth it has in the repository, because the Vite config
# copies ../../THIRD_PARTY_NOTICES.md into the bundle -- the notices the
# licences ask to travel with it. Flattened to /app that path pointed above
# the image root and the build failed on a missing file; the installer's own
# staging already reproduces the same two levels.
FROM node:20-alpine AS build
WORKDIR /src/apps/trainer_ui
COPY apps/trainer_ui/package.json package.json
COPY apps/trainer_ui/package-lock.json package-lock.json
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY THIRD_PARTY_NOTICES.md /src/THIRD_PARTY_NOTICES.md
COPY apps/trainer_ui .
RUN npm run build

# -- Serve stage --
FROM nginx:alpine
COPY --from=build /src/apps/trainer_ui/dist /usr/share/nginx/html
# Route all non-file requests to index.html (SPA support) and proxy API
COPY apps/trainer_ui/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
