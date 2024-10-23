document.addEventListener('DOMContentLoaded', function() {
    let blocks = document.querySelectorAll('pre code');
    blocks.forEach((block) => {
      let button = document.createElement('button');
      button.innerHTML = 'Copy';
      button.classList.add('copy-button');
      block.parentNode.appendChild(button);
      button.addEventListener('click', () => {
        let range = document.createRange();
        range.selectNode(block);
        window.getSelection().addRange(range);
        document.execCommand('copy');
        window.getSelection().removeAllRanges();
        button.innerHTML = 'Copied!';
        setTimeout(() => { button.innerHTML = 'Copy'; }, 2000);
      });
    });
  });