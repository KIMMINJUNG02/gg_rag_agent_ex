from IPython.display import Image, display

def visualize_graph(graph):
  """
  그래프를 시각화하여 Jupyter Notebook에 표시합니다.
  
  Args:
      graph: 시각화할 그래프 객체
  """
  # PNG 생성에 네트워크 등이 필요하면 Mermaid 텍스트로 대체한다.
  try:
      display(Image(graph.get_graph().draw_mermaid_png()))
  except Exception as exc:
      print("PNG 시각화 불가:", type(exc).__name__)
      print(graph.get_graph().draw_mermaid())