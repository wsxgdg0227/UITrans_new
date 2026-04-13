import asyncio
from ui import convert_android_component_to_harmony

print('Starting conversion test...')
try:
    res = convert_android_component_to_harmony('<Button android:id="@+id/btn" android:text="Hello" />')
    print('Result:', res)
except Exception as e:
    import traceback
    traceback.print_exc()
