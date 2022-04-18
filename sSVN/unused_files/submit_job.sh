#!/bin/bash

file='main_driver.py'
echo "Bash script executed..."

# All input settings
rescale_likelihood=(1) # Correct syntax: (a b c d ... e)
rescale_prior=(1)
gradient_step=(.00001)
hessian_step=(.00001)
iterations=50
particles=100
store_history='True'
injection=('10 10') # Correct syntax: ('10 10' '12 12' ... 'a b')
sigma_injection=('0.5 0.5') #('0.5 0.5')
particle_initialization="8 12"
stochastic_perturbation=(0)
optimize_method='SVGD'

# The ugliest thing you've ever seen.
for rescale_likelihood in "${rescale_likelihood[@]}"
do
  for rescale_prior in "${rescale_prior[@]}"
  do
    for gradient_step in "${gradient_step[@]}"
    do
      for hessian_step in "${hessian_step[@]}"
      do
        for injection in "${injection[@]}"
        do
          for particle_initialization in "${particle_initialization[@]}"
          do
            for sigma_injection in "${sigma_injection[@]}"
            do
              for stochastic_perturbation in "${stochastic_perturbation[@]}"
              do
                for optimize_method in "${optimize_method}"
                do
                  flags="-rsl $rescale_likelihood -rsp $rescale_prior -gs $gradient_step -hs $hessian_step -iter $iterations
                  -p $particles -hist $store_history -inj $injection -inj_s $sigma_injection -p_init $particle_initialization
                  -s_per $stochastic_perturbation -opt $optimize_method"

                  echo $flags

                  command="python "$file" "$flags""

                  eval $command
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Bash script completed."
