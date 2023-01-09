from opt_client import Client


client = Client("gpu065", "6969")
client.get_edited_activations(["Hi im Matt"], ["decoder"], None)

client.get_edited_activations(["Hi im Matt"], ["decoder"], {"foo": "bar"})

client.get_edited_activations(["Hi im Matt"], ["decoder"], {"decoder": "zzz"})
