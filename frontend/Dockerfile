FROM nginx:alpine
COPY . /usr/share/nginx/html/
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

EXPOSE 8080

# Extract hostname from API_BASE_URL and substitute variables in template
CMD ["/bin/sh", "-c", "\
    export API_BACKEND_HOST=$(echo $API_BASE_URL | sed 's|https://||' | sed 's|http://||' | sed 's|/.*||') && \
    envsubst '${API_BASE_URL} ${API_BACKEND_HOST}' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf && \
    echo 'API_BASE_URL='$API_BASE_URL && \
    echo 'API_BACKEND_HOST='$API_BACKEND_HOST && \
    echo 'Generated nginx config:' && \
    cat /etc/nginx/conf.d/default.conf && \
    nginx -g 'daemon off;'\
    "]