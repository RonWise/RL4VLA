# Use pre-built vla-rl image as base
FROM vla-rl:latest

# Set working directory (for clarity, base image already has /workspace)
WORKDIR /workspace

# ---- RL4VLA specific installation ----
# Install additional dependencies required by RL4VLA
RUN pip install --no-cache-dir -U tyro datasets==3.3.2

# Update sapien to v3.0.0.b1 (required by ManiSkill v3, base image has 2.2.2)
RUN pip install --no-cache-dir --upgrade 'sapien==3.0.0.b1'

# Copy and install RL4VLA packages
COPY openvla /workspace/openvla
COPY ManiSkill /workspace/ManiSkill
COPY SimplerEnv /workspace/SimplerEnv

# Install packages in editable mode
RUN cd /workspace/openvla && pip install --no-cache-dir -e . && \
    cd /workspace/ManiSkill && pip install --no-cache-dir -e . && \
    cd /workspace/SimplerEnv && pip install --no-cache-dir -e .

