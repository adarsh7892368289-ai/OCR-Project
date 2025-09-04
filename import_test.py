import sys; sys.path.insert(0, '.')

components = [
    'src.postprocessing.text_corrector',
    'src.postprocessing.confidence_filter', 
    'src.postprocessing.layout_analyzer',
    'src.postprocessing.result_formatter'
]

for component in components:
    try:
        __import__(component)
        print(f'✅ {component}')
    except Exception as e:
        print(f'❌ {component}: {e}')