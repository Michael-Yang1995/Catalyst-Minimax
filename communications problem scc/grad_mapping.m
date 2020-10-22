[test_grad_x1, test_grad_y1] = grad(output_svrg.x, output_svrg.y, sigma, beta, lambda);
norm(output_svrg.y - reshape(proj(output_svrg.y + 0.1*test_grad_y1, a),n,1))
norm(test_grad_x1,2)
