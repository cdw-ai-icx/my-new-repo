# 🚀 FastMCP Streamable HTTP Integration Deployment Guide

## 🔥 What This Integration Delivers

Your CDW Explore AdvBot now has **DIRECT** access to Epic FHIR healthcare data through FastMCP 2.6.1 with **streamable HTTP transport**:

- **⚡ FastMCP 2.6.1**: Latest version with streamable HTTP optimizations
- **🌐 Streamable HTTP**: Real-time bidirectional communication over HTTP
- **🏥 Healthcare Intelligence**: Real-time patient data, appointments, medications
- **🧠 Smart Detection**: Automatically detects healthcare queries and routes to MCP
- **🔄 Fallback Ready**: Graceful error handling with retry logic
- **📊 Enhanced Responses**: Enriches LangChain responses with live healthcare data

## 📁 File Structure

```
your-lambda-project/
├── lambda_function.py      # Updated with minimal MCP integration
├── mcp_tools.py           # NEW - FastMCP streamable HTTP client
├── requirements.txt       # Updated with FastMCP 2.6.1 + httpx
└── deployment/
    ├── package.zip        # Your Lambda deployment package
    └── environment.yml    # Environment variables
```

## 🚀 Deployment Steps

### 1. Environment Variables

Add these environment variables to your Lambda:

```bash
# Required - Your Epic MCP Server URL (streamable HTTP)
EPIC_MCP_SERVER_URL=https://epicmcp.cdwaws.com/mcp/

# Optional - MCP configuration
MCP_TIMEOUT_SECONDS=30
MCP_MAX_RETRIES=3

# Existing variables (keep these)
AWS_REGION=us-east-1
CHAT_HISTORY_TABLE=cdw-explore-advbot-chat-history
```

### 2. Lambda Configuration Updates

**Memory**: Increase to **768MB** (streamable HTTP + httpx needs more memory)
**Timeout**: Set to **45 seconds** (allows for MCP tool execution + retry logic)
**Runtime**: Python 3.11+
**Network**: Ensure outbound HTTPS access to `epicmcp.cdwaws.com`

### 3. Deploy the Code

```bash
# Install dependencies
pip install -r requirements.txt -t ./

# Package for Lambda
zip -r deployment.zip . -x "*.pyc" "__pycache__/*" "test_*" "*.git*"

# Upload to Lambda (replace with your function name)
aws lambda update-function-code \
    --function-name cdw-explore-advbot \
    --zip-file fileb://deployment.zip
```

## 🧪 Testing Your Integration

### Test 1: General Query (Should NOT use MCP)
```json
{
  "body": {
    "message": "What are the latest trends in cloud computing?",
    "session_id": "test-session-1"
  }
}
```

**Expected**: Normal CDW advisory response, `used_mcp: false`

### Test 2: Healthcare Query (Should use MCP)
```json
{
  "body": {
    "message": "Can you help me look up patient information in Epic?",
    "session_id": "test-session-2"
  }
}
```

**Expected**: Healthcare-aware response, `used_mcp: true`, MCP tools listed

### Test 3: Direct Patient Lookup (MCP Tools Active)
```json
{
  "body": {
    "message": "Find patient with ID e0w0LEDCYtfckT6N.CkJKCw3",
    "session_id": "test-session-3"
  }
}
```

**Expected**: Actual patient data retrieved via FastMCP

## 🔍 Response Format Changes

Your responses now include streamable HTTP metadata:

```json
{
  "response": "Here's the patient information...",
  "session_id": "test-session",
  "bot_name": "cdw-explore-advbot-mcp-enabled",
  "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
  "mcp_enabled": true,
  "mcp_transport": "streamable_http",
  "fastmcp_version": "2.6.1",
  "used_mcp": true,
  "mcp_result": {
    "success": true,
    "transport": "streamable_http",
    "server_ready": true,
    "tools_available": [
      "get_patient_by_id",
      "match_patient", 
      "search_patients",
      "get_patient_appointments",
      "get_patient_complete_profile",
      "get_patient_medications_detailed"
    ],
    "response_type": "healthcare_query"
  }
}
```

## 🛡️ Error Handling & Fallbacks

The integration is designed to be **bulletproof**:

1. **MCP Server Down**: Bot continues with standard responses + warning
2. **Network Issues**: Graceful timeout, fallback to general advice
3. **Invalid Queries**: Smart error messages guide users to correct format
4. **Tool Failures**: Individual tool errors don't break the conversation

## 🔧 Available MCP Tools

Your Epic MCP server exposes these tools:

- **`get_patient_by_id`**: Direct patient lookup
- **`match_patient`**: Find patients by demographics  
- **`search_patients`**: Search patient database
- **`get_patient_appointments`**: Appointment history/upcoming
- **`get_patient_complete_profile`**: Comprehensive patient view
- **`get_patient_medications_detailed`**: Medication history with details

## 📈 Monitoring & Logs

Look for these log patterns:

```
🌐 MCP Tools Manager initialized for streamable HTTP transport: https://epicmcp.cdwaws.com/mcp/
✅ MCP initialized with 6 tools via streamable HTTP: get_patient_by_id, match_patient...
🔧 MCP tool 'get_patient_by_id' executed successfully via streamable HTTP
⏰ MCP tool 'search_patients' timed out after 30s: Connection timeout
🔌 MCP connection failed for 'get_patient_by_id': Failed to connect to MCP server
🏥 HEALTHCARE DATA ACCESS: Retrieved relevant information...
```

## 🚀 Next Level Enhancements

Once this is working, you can easily extend:

1. **Add More MCP Servers**: Connect to multiple healthcare systems
2. **Custom Tools**: Create CDW-specific MCP tools
3. **Caching Layer**: Cache frequent patient lookups
4. **Advanced Routing**: Smarter query detection and tool selection

## 🆘 Troubleshooting

### Common Issues:

**1. "MCP tools not available"**
- Check `EPIC_MCP_SERVER_URL` environment variable
- Verify server is accessible from Lambda VPC
- Check Lambda memory settings

**2. "Tool execution timeout"**
- Increase Lambda timeout
- Check Epic server response times
- Verify network connectivity

**3. "Invalid patient ID format"**
- Epic patient IDs are long (20+ characters)
- Format: alphanumeric with periods/special chars
- Example: `e0w0LEDCYtfckT6N.CkJKCw3`

## 💡 Pro Tips

1. **Healthcare Keywords**: The bot auto-detects healthcare queries using keywords like "patient", "appointment", "medication", "epic", etc.

2. **Privacy First**: Patient data is never stored in DynamoDB conversation history - only processed in real-time

3. **Scalable**: FastMCP client handles connection pooling automatically

4. **Future-Ready**: Easy to add new MCP servers or tools without changing Lambda code

---

**🎯 Bottom Line**: Your bot now has superpowers! It can seamlessly blend technology consulting with real-time healthcare data access, making it incredibly valuable for healthcare IT scenarios.