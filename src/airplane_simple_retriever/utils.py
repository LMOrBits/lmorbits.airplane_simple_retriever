from langchain_core.messages import AnyMessage


def message_converter(message: AnyMessage):
  return {"role": message.type, "content": message.content}


