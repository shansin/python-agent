import ollama
import time
import json
import argparse
import os
from datetime import datetime


class ModelTester:
    def __init__(self, host=None):
        #self.test_message = "Just say 'Hello, this is a test response.'"
        self.test_message = "Write a story about a cart"
        self.test_results = {}
        self.start_time = time.time()
        
        if host:
            if hasattr(ollama, 'Client'):
                self.client = ollama.Client(host=host)
            else:
                # Fallback for older versions or if Client not found
                os.environ["OLLAMA_HOST"] = host
                self.client = ollama
        else:
            self.client = ollama

    def get_chat_models(self):
        """Get list of chat models (exclude embeddings)"""
        try:
            response = self.client.list()
            chat_models = []

            for model in response.models:
                model_name = str(model.model)
                details = model.details
                family = str(details.family).lower() if details else ""

                # Skip embedding models
                if "embed" in model_name.lower() or "embed" in family:
                    print(f"⏭️  Skipping embedding model: {model_name}")
                    continue

                chat_models.append(model_name)

            return chat_models
        except Exception as e:
            print(f"❌ Error getting models: {e}")
            return []

    def test_model(self, model_name, timeout_seconds=900):  # 15 minutes
        """Test a single model with timeout"""
        print(f"\n🧪 Testing model: {model_name}")
        print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Timeout set to: {timeout_seconds // 60} minutes")

        try:
            # Use simple message format for all models to be safe
            messages = [{"role": "user", "content": self.test_message}]

            # Start timing
            model_start = time.time()

            # Stream response with timeout
            response = ""
            thinking_timeout = 300  # 5 minutes before showing "still thinking" message
            last_thinking_message = time.time()

            print("💭 Thinking...")

            stream = self.client.chat(model=model_name, messages=messages, stream=True)

            total_tokens = 0
            eval_duration_ns = 0

            for chunk in stream:
                current_time = time.time()
                
                # These fields only exist in some chunks
                if 'eval_count' in chunk:
                    total_tokens = chunk['eval_count']  # This is cumulative
                if 'eval_duration' in chunk:
                    eval_duration_ns = chunk['eval_duration']  # This is total time

                # Check for timeout
                if current_time - model_start > timeout_seconds:
                    print(
                        f"⏰ TIMEOUT: {model_name} exceeded {timeout_seconds // 60} minutes"
                    )
                    self.test_results[model_name] = {
                        "status": "TIMEOUT",
                        "response_time": f">{timeout_seconds // 60}min",
                        "response": None,
                        "error": f"Timeout after {timeout_seconds // 60} minutes",
                    }
                    return False

                # Show "still thinking" message every 5 minutes
                if current_time - last_thinking_message > thinking_timeout:
                    elapsed = int((current_time - model_start) / 60)
                    print(f"💭 Still thinking... ({elapsed} minutes elapsed)")
                    last_thinking_message = current_time

                if "message" in chunk and "content" in chunk["message"]:
                    chunk_content = chunk["message"]["content"]
                    if chunk_content:
                        if not response:  # First content received
                            elapsed = int((time.time() - model_start) / 60)
                            print(f"🎯 First response received at {elapsed} minutes!")
                        response += chunk_content

            # Successfully got response
            elapsed_time = int(time.time() - model_start)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60

            print(f"✅ {model_name} responded in {minutes}m {seconds}s")
            print(
                f"📝 Response: {response[:100]}..."
                if len(response) > 100
                else f"📝 Response: {response}"
            )

            # Calculate tokens per second
            tokens_per_sec = 0
            if eval_duration_ns > 0 and total_tokens > 0:
                time_in_seconds = eval_duration_ns / 1_000_000_000
                tokens_per_sec = total_tokens / time_in_seconds

            self.test_results[model_name] = {
                "status": "SUCCESS",
                "response_time": f"{minutes}m {seconds}s",
                "tokens_per_second": f"{tokens_per_sec:.2f}",
                "total_tokens": total_tokens,
                "response": response,
                "error": None,
            }
            return True

        except Exception as e:
            elapsed_time = int(time.time() - model_start)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60

            error_msg = str(e)
            print(f"❌ {model_name} failed after {minutes}m {seconds}s: {error_msg}")

            self.test_results[model_name] = {
                "status": "ERROR",
                "response_time": f"{minutes}m {seconds}s",
                "response": None,
                "error": error_msg,
            }
            return False

    def test_all_models(self):
        """Test all available models"""
        print("🚀 Starting comprehensive model test")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Get models
        models = self.get_chat_models()
        if not models:
            print("❌ No models found to test")
            return

        print(f"📋 Found {len(models)} models to test:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")

        print("=" * 60)

        # Test each model
        successful_tests = 0
        failed_tests = 0

        for i, model_name in enumerate(models, 1):
            print(f"\n📍 Progress: {i}/{len(models)} models tested")
            print(
                f"📊 Current success rate: {successful_tests}/{i - 1} ({successful_tests / (max(i - 1, 1)) * 100:.1f}%)"
            )

            success = self.test_model(model_name)
            if success:
                successful_tests += 1
            else:
                failed_tests += 1

        # Final summary
        total_elapsed = int(time.time() - self.start_time)
        total_minutes = total_elapsed // 60

        output_lines = []
        output_lines.append("\n" + "=" * 60)
        output_lines.append("🏁 COMPREHENSIVE TEST COMPLETE")
        output_lines.append(f"🕐 Completed at: {datetime.now().strftime('%H:%M:%S')}")
        output_lines.append(f"⏱️  Total time: {total_minutes} minutes")
        output_lines.append("=" * 60)
        output_lines.append(f"📊 SUMMARY:")
        output_lines.append(f"  ✅ Successful: {successful_tests}/{len(models)}")
        output_lines.append(f"  ❌ Failed: {failed_tests}/{len(models)}")
        # Handle zero division if len(models) is 0, though we return early if not models earlier
        success_rate = successful_tests / len(models) * 100 if len(models) > 0 else 0
        output_lines.append(f"  📈 Success rate: {success_rate:.1f}%")
        output_lines.append("=" * 60)

        # Show detailed results
        output_lines.append("\n📋 DETAILED RESULTS:")
        for model_name, result in self.test_results.items():
            status_emoji = (
                "✅"
                if result["status"] == "SUCCESS"
                else "❌"
                if result["status"] == "ERROR"
                else "⏰"
            )
            output_lines.append(f"\n{status_emoji} {model_name}")
            output_lines.append(f"   Status: {result['status']}")
            output_lines.append(f"   Time: {result['response_time']}")
            if result.get('tokens_per_second'):
                output_lines.append(f"   Tokens/sec: {result['tokens_per_second']}")
            if result.get('total_tokens'):
                output_lines.append(f"   Total tokens: {result['total_tokens']}")
            if result["error"]:
                output_lines.append(f"   Error: {result['error']}")
            if result["response"]:
                response_preview = (
                    result["response"][:150] + "..."
                    if len(result["response"]) > 150
                    else result["response"]
                )
                output_lines.append(f"   Response: {response_preview}")

        final_output = "\n".join(output_lines)
        print(final_output)

        # Write to file
        os.makedirs("output", exist_ok=True)
        filename = datetime.now().strftime("model_benchmark_%Y_%m_%d__%H_%M.txt")
        filepath = os.path.join("output", filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final_output)
            print(f"\n💾 Results saved to: {filepath}")
        except Exception as e:
            print(f"\n❌ Failed to save results to file: {e}")

        return self.test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local Ollama models")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL (e.g., http://localhost:11434)")
    args = parser.parse_args()

    tester = ModelTester(host=args.host)
    results = tester.test_all_models()
