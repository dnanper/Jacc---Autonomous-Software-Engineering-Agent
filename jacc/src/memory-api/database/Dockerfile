# ====================================================
# Memory API - PostgreSQL with pgvector Dockerfile
# ====================================================
# Based on the official pgvector image for PostgreSQL 16
# Includes extensions: vector, pg_trgm, btree_gin

FROM pgvector/pgvector:pg16

# ====================================================
# Create init script directory
# ====================================================
RUN mkdir -p /docker-entrypoint-initdb.d

# ====================================================
# Enable required PostgreSQL extensions
# ====================================================
# These extensions are created when the database is first initialized
# - vector: pgvector for vector similarity search
# - pg_trgm: trigram matching for fuzzy text search
# - btree_gin: GIN index support for scalar types
RUN echo "-- Enable pgvector extension for vector similarity search" > /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "CREATE EXTENSION IF NOT EXISTS vector;" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "-- Enable pg_trgm for fuzzy text matching" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "CREATE EXTENSION IF NOT EXISTS pg_trgm;" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "-- Enable btree_gin for GIN indexes on scalar types" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "CREATE EXTENSION IF NOT EXISTS btree_gin;" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "" >> /docker-entrypoint-initdb.d/01-extensions.sql && \
    echo "-- Extension setup complete" >> /docker-entrypoint-initdb.d/01-extensions.sql

# ====================================================
# PostgreSQL Configuration Optimization
# ====================================================
# Optimize PostgreSQL for vector operations and memory workloads
RUN echo "" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "# Memory API - Vector operations optimization" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "shared_preload_libraries = 'vector'" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "# Increase work_mem for vector operations" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "# work_mem = 256MB" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "# Increase maintenance_work_mem for index creation" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "# maintenance_work_mem = 512MB" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "# Increase effective_cache_size for query planning" >> /usr/share/postgresql/postgresql.conf.sample && \
    echo "# effective_cache_size = 2GB" >> /usr/share/postgresql/postgresql.conf.sample

# ====================================================
# Expose PostgreSQL port
# ====================================================
EXPOSE 5432
