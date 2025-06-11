#!/usr/bin/env python3
"""
Test script to verify MCP integration is working correctly
"""
import asyncio
import json
from mcp_tools import get_available_tools, call_mcp_tool, get_mcp_tools_context

async def test_mcp_connection():
    """Test basic MCP server connection and tool listing"""
    print("🔍 Testing MCP server connection...")
    
    try:
        # Test 1: Get available tools
        print("\n📋 Testing tool listing...")
        tools = await get_available_tools()
        
        if tools:
            print(f"✅ Found {len(tools)} MCP tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        else:
            print("❌ No tools found or connection failed")
            return False
        
        # Test 2: Test context generation for healthcare query
        print("\n🏥 Testing healthcare query detection...")
        test_query = "Can you check my past appointments? My patient ID is erXuFYUfucBZaryVksYEcMg3"
        context_result = await get_mcp_tools_context(test_query)
        
        print(f"Context used MCP: {context_result.get('used_mcp', False)}")
        print(f"Context message: {context_result.get('context', '')[:300]}...")
        
        # Test 3: Try calling a tool if available
        if tools:
            print(f"\n🔧 Testing tool execution...")
            first_tool = tools[0]
            tool_name = first_tool['name']
            
            # Try calling with generic test parameters (let tool schema determine what's needed)
            test_args = {}
            
            # Look at tool schema to determine appropriate test parameters
            schema = first_tool.get('input_schema', {})
            if schema and 'properties' in schema:
                for param_name, param_def in schema['properties'].items():
                    if 'patient' in param_name.lower() and 'id' in param_name.lower():
                        test_args[param_name] = "erXuFYUfucBZaryVksYEcMg3"
                    elif param_name.lower() in ['id', 'patient_id', 'patientid']:
                        test_args[param_name] = "erXuFYUfucBZaryVksYEcMg3"
            
            if not test_args:
                print(f"  No obvious parameters found for {tool_name}, skipping tool test")
            else:
                result = await call_mcp_tool(tool_name, test_args)
                
                print(f"Tool {tool_name} result:")
                print(f"  Success: {result.get('success', False)}")
                if result.get('success'):
                    print(f"  Result: {result.get('result', [])[:2]}...")  # First 2 lines
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_lambda_simulation():
    """Simulate a Lambda function call"""
    print("\n🚀 Testing Lambda function simulation...")
    
    # Import our lambda function
    import lambda_function
    
    # Create test event
    test_event = {
        "body": json.dumps({
            "message": "Hi there, could you check and see what my past appointment dates were? Make sure these are pulled from our Epic system! my patient id is erXuFYUfucBZaryVksYEcMg3",
            "session_id": "test-session-mcp"
        })
    }
    
    # Mock context
    class MockContext:
        aws_request_id = "test-request-123"
    
    try:
        result = lambda_function.lambda_handler(test_event, MockContext())
        print(f"✅ Lambda function completed")
        print(f"Status Code: {result.get('statusCode')}")
        
        if result.get('statusCode') == 200:
            body = json.loads(result.get('body', '{}'))
            response = body.get('response', '')
            used_mcp = body.get('used_mcp', False)
            
            print(f"Used MCP: {used_mcp}")
            print(f"Response: {response[:300]}...")
            
            if not used_mcp:
                print("⚠️ WARNING: MCP tools were not used for patient query!")
            
            # Check for made-up data indicators
            made_up_indicators = [
                "I don't have access",
                "I can't access",
                "privacy concerns",
                "cannot provide",
                "unable to access"
            ]
            
            response_lower = response.lower()
            for indicator in made_up_indicators:
                if indicator in response_lower:
                    print(f"⚠️ WARNING: Response contains '{indicator}' - may be refusing access")
        
        return result.get('statusCode') == 200
        
    except Exception as e:
        print(f"❌ Lambda simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🧪 Starting MCP Integration Tests\n")
    
    # Test 1: Basic MCP connection
    mcp_test_passed = await test_mcp_connection()
    
    # Test 2: Lambda simulation
    lambda_test_passed = await test_lambda_simulation()
    
    print(f"\n📊 Test Results:")
    print(f"  MCP Connection: {'✅ PASS' if mcp_test_passed else '❌ FAIL'}")
    print(f"  Lambda Simulation: {'✅ PASS' if lambda_test_passed else '❌ FAIL'}")
    
    if mcp_test_passed and lambda_test_passed:
        print(f"\n🎉 All tests passed! MCP integration is working.")
    else:
        print(f"\n⚠️ Some tests failed. Check the logs above.")

if __name__ == "__main__":
    asyncio.run(main())