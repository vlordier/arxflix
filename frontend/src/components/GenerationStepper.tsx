import { Step, type StepItem, Stepper, useStepper } from "@/components/stepper";
import { Button } from "@/components/ui/button";

/**
 * Define the steps for the stepper component
 */
const steps: StepItem[] = [
  { label: "Step 1" },
  { label: "Step 2" },
  { label: "Step 3" },
];

/**
 * Component representing the stepper with multiple steps
 * @returns {JSX.Element} GenerationStepper component
 */
export default function GenerationStepper(): JSX.Element {
  return (
    <div className="flex w-full flex-col gap-4">
      <Stepper orientation="vertical" initialStep={0} steps={steps}>
        {steps.map((stepProps, index) => (
          <Step key={stepProps.label} {...stepProps}>
            <StepContent index={index} />
            <StepButtons />
          </Step>
        ))}
        <FinalStep />
      </Stepper>
    </div>
  );
}

/**
 * Component representing the content for each step
 * @param {number} index - Index of the current step
 * @returns {JSX.Element} StepContent component
 */
const StepContent: React.FC<{ index: number }> = ({ index }): JSX.Element => (
  <div className="h-40 flex items-center justify-center my-4 border bg-secondary text-primary rounded-md">
    <h1 className="text-xl">Step {index + 1}</h1>
  </div>
);

/**
 * Component for rendering the buttons to navigate between steps
 * @returns {JSX.Element} StepButtons component
 */
const StepButtons: React.FC = (): JSX.Element => {
  const { nextStep, prevStep, isLastStep, isOptionalStep, isDisabledStep } = useStepper();

  return (
    <div className="w-full flex gap-2 mb-4">
      <Button
        disabled={isDisabledStep}
        onClick={prevStep}
        size="sm"
        variant="secondary"
      >
        Prev
      </Button>
      <Button size="sm" onClick={nextStep}>
        {isLastStep ? "Finish" : isOptionalStep ? "Skip" : "Next"}
      </Button>
    </div>
  );
};

/**
 * Component for rendering the final step and reset button
 * @returns {JSX.Element | null} FinalStep component
 */
const FinalStep: React.FC = (): JSX.Element | null => {
  const { hasCompletedAllSteps, resetSteps } = useStepper();

  if (!hasCompletedAllSteps) {
    return null;
  }

  return (
    <>
      <div className="h-40 flex items-center justify-center border bg-secondary text-primary rounded-md">
        <h1 className="text-xl">Woohoo! All steps completed! 🎉</h1>
      </div>
      <div className="w-full flex justify-end gap-2">
        <Button size="sm" onClick={resetSteps}>
          Reset
        </Button>
      </div>
    </>
  );
};
