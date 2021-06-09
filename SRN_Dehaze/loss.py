def get_loss(outputs,targets,loss_funcs,times,weights):
  loss = 0
  losses = []
  for i in range(times):
    weight = weights[i]
    out = outputs[i]
    target = targets[i]

    for func in loss_funcs:
      loss += weight * func(out, target)

    losses.append(loss)
  return loss, losses
      

  
